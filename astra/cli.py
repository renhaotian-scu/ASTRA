
import argparse
import json
import torch
from .model import build_astra
from .deform_backend import check_deform_backend

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Report architecture and parameter counts")
    smoke = sub.add_parser("smoke", help="Check random-weight inference and deployment pruning")
    for p in (inspect, smoke):
        p.add_argument("--backend", choices=["pytorch", "torchvision", "mmcv", "auto"], default="pytorch")
    smoke.add_argument("--height", type=int, default=128)
    smoke.add_argument("--width", type=int, default=128)
    smoke.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = build_astra(backend=args.backend).eval()
    deployed = model.to_deploy()
    report = {
        "core_version": "0.1.3",
        "backend": model.backend,
        "parameters_with_auxiliary": sum(p.numel() for p in model.parameters()),
        "parameters_deployed": sum(p.numel() for p in deployed.parameters()),
        "scope": "Untrained core model; no paper-accuracy or historical-checkpoint claim",
    }
    if args.command == "smoke":
        report["operator"] = check_deform_backend(args.backend, args.device)
        model.to(args.device)
        deployed.to(args.device)
        x = torch.randn(1, 3, args.height, args.width, device=args.device)
        with torch.no_grad():
            y = model(x)[0]
            z = deployed(x)[0]
        report.update(input_shape=list(x.shape), prediction_shape=list(y.shape),
                      finite_predictions=bool(torch.isfinite(y).all()),
                      max_eval_deploy_difference=float((y-z).abs().max()))
        if not report["finite_predictions"] or report["max_eval_deploy_difference"] > 1e-5:
            raise RuntimeError("Core forward/deployment check failed")
    print(json.dumps(report, indent=2))
    return report
