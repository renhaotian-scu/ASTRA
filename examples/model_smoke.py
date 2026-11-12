"""Exercise the real core graph and auxiliary-pruned deployment conversion."""
import argparse
import json
import torch
from astra import build_astra, check_deform_backend


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--device', default='cpu')
    p.add_argument('--backend', choices=['pytorch', 'torchvision', 'mmcv', 'auto'], default='pytorch')
    a = p.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    health = check_deform_backend(a.backend, a.device)
    model = build_astra(backend=a.backend).to(a.device).eval()
    x = torch.randn(1, 3, a.size, a.size, device=a.device)
    deployed = model.to_deploy()
    with torch.no_grad():
        y = model(x)[0]
        z = deployed(x)[0]
    error = float((y - z).abs().max())
    if not torch.isfinite(y).all() or error > 1e-5:
        raise RuntimeError('Own-graph deployment equivalence check failed')
    print(json.dumps({'operator': health, 'prediction_shape': list(y.shape),
        'finite_predictions': bool(torch.isfinite(y).all()),
        'training_graph_parameters': sum(v.numel() for v in model.parameters()),
        'deployment_graph_parameters': sum(v.numel() for v in deployed.parameters()),
        'max_eval_deploy_difference': error,
        'scope': 'Random untrained weights; not paper accuracy or historical checkpoint parity'}, indent=2))


if __name__ == '__main__':
    main()
