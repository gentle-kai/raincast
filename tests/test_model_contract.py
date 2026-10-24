"""Check modelling boundaries using random tensors only."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_tp_radar_multitask import MultiTaskUNet, losses, prepare_modalities


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_initial_regression_equals_background_and_residual_is_bounded(self):
        model = MultiTaskUNet(base=8)
        background = torch.rand(1, 3, 16, 16)
        features = torch.rand(1, 15, 16, 16)
        pred, logits = model(features, background)
        torch.testing.assert_close(pred, background)
        self.assertEqual(tuple(logits.shape), (1, 3, 5, 16, 16))
        with torch.no_grad():
            model.regression.bias.fill_(100)
        pred, _ = model(features, background)
        self.assertTrue(((pred - background).abs() <= .350001).all())

    def test_full_modality_drop_zeros_data_background_and_flags(self):
        features = torch.ones(2, 12, 16, 16)
        background = torch.ones(2, 3, 16, 16)
        prepared, anchor = prepare_modalities(features, background, True, 1, 1, 1)
        self.assertEqual(tuple(prepared.shape), (2, 15, 16, 16))
        self.assertEqual(int(torch.count_nonzero(prepared)), 0)
        self.assertEqual(int(torch.count_nonzero(anchor)), 0)
        self.assertTrue((features == 1).all())

    def test_masked_loss_has_finite_gradients_and_ignores_masked_target(self):
        model = MultiTaskUNet(base=8)
        background = torch.ones(1, 3, 16, 16)
        pred, logits = model(torch.rand(1, 15, 16, 16), background)
        target = torch.log1p(torch.rand_like(background) * 60)
        valid = torch.ones_like(background)
        valid[..., :8, :] = 0
        total, _ = losses(pred, logits, target, valid)
        changed = target.clone()
        changed[..., :8, :] = 0
        altered, _ = losses(pred, logits, changed, valid)
        torch.testing.assert_close(total, altered)
        total.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients))


if __name__ == '__main__':
    unittest.main()
