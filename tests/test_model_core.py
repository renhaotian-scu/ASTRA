"""Real core-model and sampling checks; synthetic inputs do not measure accuracy."""
import unittest
import torch
from torch.nn import functional as F
from astra import build_astra
from astra.deform_backend import modulated_deform_conv2d_reference as deform


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)

    def test_zero_offsets_match_grouped_convolution(self):
        for groups, deform_groups in [(1, 1), (2, 1), (2, 2)]:
            x = torch.randn(2, 4, 7, 8, dtype=torch.double)
            w = torch.randn(6, 4 // groups, 3, 3, dtype=torch.double)
            b = torch.randn(6, dtype=torch.double)
            offsets = torch.zeros(2, deform_groups * 18, 7, 8, dtype=torch.double)
            mask = torch.ones(2, deform_groups * 9, 7, 8, dtype=torch.double)
            y = deform(x, offsets, mask, w, b, 1, 1, 1, groups, deform_groups)
            torch.testing.assert_close(y, F.conv2d(x, w, b, padding=1, groups=groups), atol=1e-12, rtol=1e-12)

    def test_nonzero_offsets_and_zero_boundary_against_manual_bilinear(self):
        x = torch.tensor([[[[1., 2.], [3., 4.]]]], dtype=torch.double)
        offset = torch.empty(1, 2, 2, 2, dtype=torch.double)
        offset[:, 0] = .5
        offset[:, 1] = .25
        y = deform(x, offset, torch.ones(1, 1, 2, 2, dtype=torch.double),
                   torch.ones(1, 1, 1, 1, dtype=torch.double), None, 1, 0, 1)
        # Pixel-center sampling at (dy=.5,dx=.25), with samples outside set to zero.
        expected = torch.tensor([[[[2.25, 2.25], [1.625, 1.5]]]], dtype=torch.double)
        torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-12)

    def test_modulation_does_not_scale_bias(self):
        x = torch.randn(1, 2, 5, 5, dtype=torch.double)
        w = torch.randn(3, 2, 3, 3, dtype=torch.double)
        b = torch.randn(3, dtype=torch.double)
        y = deform(x, torch.zeros(1, 18, 5, 5, dtype=torch.double),
                   torch.full((1, 9, 5, 5), .5, dtype=torch.double), w, b, 1, 1, 1)
        expected = .5 * F.conv2d(x, w, padding=1) + b.view(1, 3, 1, 1)
        torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-12)

    def test_sampling_offset_mask_input_and_weight_gradcheck(self):
        x = torch.randn(1, 1, 2, 2, dtype=torch.double, requires_grad=True)
        offset = torch.full((1, 2, 2, 2), .23, dtype=torch.double, requires_grad=True)
        mask = torch.full((1, 1, 2, 2), .73, dtype=torch.double, requires_grad=True)
        w = torch.randn(1, 1, 1, 1, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda xx, oo, mm, ww: deform(xx, oo, mm, ww, None, 1, 0, 1),
            (x, offset, mask, w), eps=1e-6, atol=1e-5, rtol=1e-3))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_regular_eval_and_pruned_deployment_are_identical(self):
        model = build_astra(backend="pytorch").eval()
        deploy = model.to_deploy()
        self.assertTrue(model.auxiliary)
        self.assertFalse(deploy.auxiliary)
        self.assertEqual(sum(p.numel() for p in deploy.parameters()), 20252530)
        with torch.no_grad():
            x = torch.randn(1, 3, 128, 192)
            y = model(x)[0]
            z = deploy(x)[0]
        self.assertEqual(tuple(y.shape), (1, 300, 5))
        self.assertTrue(bool(torch.isfinite(y).all()))
        torch.testing.assert_close(y, z, atol=0, rtol=0)


    def test_input_constraints_fail_with_a_clear_error(self):
        model = build_astra(backend="pytorch").eval()
        with self.assertRaisesRegex(ValueError, "divisible by 64"):
            model(torch.zeros(1, 3, 128, 160))


if __name__ == "__main__":
    unittest.main()
