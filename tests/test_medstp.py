from __future__ import annotations

import unittest

import torch

from medstp import select_medstp
from medstp.selector import resolve_grid


class MedSTPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.tokens = torch.randn(144, 64)

    def test_exact_budget_and_original_order(self):
        for keep in (0.01, 0.05, 0.1, 0.5, 0.95):
            with self.subTest(keep=keep):
                result = select_medstp(self.tokens, keep, grid_hw=(12, 12))
                self.assertEqual(result.indices.numel(), int(keep * len(self.tokens)))
                self.assertEqual(result.indices.unique().numel(), result.indices.numel())
                self.assertTrue(torch.equal(result.indices, result.indices.sort().values))
                self.assertTrue(torch.equal(result.pruned_tokens, self.tokens[result.indices]))

    def test_merged_grid_matches_explicit_grid(self):
        a = select_medstp(self.tokens, image_grid_thw=torch.tensor([[1, 24, 24]]))
        b = select_medstp(self.tokens, grid_hw=(12, 12))
        self.assertTrue(torch.equal(a.indices, b.indices))

    def test_direct_grid_metadata_fallback(self):
        self.assertEqual(resolve_grid(144, torch.tensor([1, 12, 12]), 2, None), (12, 12))

    def test_rng_state_and_input_are_preserved(self):
        before = self.tokens.clone()
        state = torch.random.get_rng_state().clone()
        a = select_medstp(self.tokens, grid_hw=(12, 12))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(before, self.tokens))
        b = select_medstp(self.tokens, grid_hw=(12, 12))
        self.assertTrue(torch.equal(a.indices, b.indices))

    def test_environment_cannot_change_configuration(self):
        import os
        from unittest.mock import patch
        expected = select_medstp(self.tokens, grid_hw=(12, 12))
        with patch.dict(os.environ, {"PRUNESID_PCA_SEED": "99", "PCA_RESIDUAL_RANK_RATIO": "0.75", "PCA_RESIDUAL_BUDGET_MODE": "energy"}):
            actual = select_medstp(self.tokens, grid_hw=(12, 12))
        self.assertTrue(torch.equal(expected.indices, actual.indices))

    def test_zero_full_and_empty(self):
        self.assertEqual(select_medstp(self.tokens, 0).indices.numel(), 0)
        self.assertTrue(torch.equal(select_medstp(self.tokens, 1).pruned_tokens, self.tokens))
        self.assertEqual(select_medstp(torch.empty(0, 64)).indices.numel(), 0)

    def test_small_and_narrow_grids(self):
        for h, w in ((2, 4), (1, 32), (2, 20), (4, 7)):
            with self.subTest(grid=(h, w)):
                tokens = torch.randn(h * w, 32)
                out = select_medstp(tokens, .5, grid_hw=(h, w))
                self.assertEqual(out.indices.numel(), int(.5 * h * w))

    def test_degenerate_tokens(self):
        for tokens in (torch.zeros(64, 32), torch.ones(64, 32)):
            out = select_medstp(tokens, grid_hw=(8, 8))
            self.assertEqual(out.indices.numel(), 6)
            self.assertEqual(out.indices.unique().numel(), 6)

    def test_dtype_is_preserved(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            out = select_medstp(self.tokens.to(dtype), grid_hw=(12, 12))
            self.assertEqual(out.pruned_tokens.dtype, dtype)

    def test_beta_zero_is_supported(self):
        out = select_medstp(self.tokens, beta=0, grid_hw=(12, 12))
        self.assertEqual(out.indices.numel(), 14)

    def test_invalid_inputs_fail_explicitly(self):
        for kwargs in ({}, {"grid_hw": (10, 10)}, {"image_grid_thw": torch.tensor([[2, 12, 12]])}, {"image_grid_thw": torch.tensor([[1, 24, 24], [1, 24, 24]])}, {"grid_hw": (12, 12), "image_grid_thw": torch.tensor([1, 24, 24])}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_medstp(self.tokens, **kwargs)
        for keep in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                select_medstp(self.tokens, keep, grid_hw=(12, 12))
        for beta in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                select_medstp(self.tokens, beta=beta, grid_hw=(12, 12))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        tokens = self.tokens.cuda()
        result = select_medstp(tokens, grid_hw=(12, 12))
        self.assertEqual(result.indices.device, tokens.device)
        self.assertEqual(result.pruned_tokens.device, tokens.device)
        self.assertTrue(torch.equal(result.pruned_tokens, tokens[result.indices]))


if __name__ == "__main__":
    unittest.main()
