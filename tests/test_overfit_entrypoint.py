import unittest

from tools.overfit_rectangular_conditional_jit import _resolve_overfit_batch_size


class OverfitEntrypointTests(unittest.TestCase):
    def test_default_overfit_batch_size_uses_configured_batch_size(self) -> None:
        self.assertEqual(_resolve_overfit_batch_size(None, configured_batch_size=2, subset_size=8), 2)

    def test_default_overfit_batch_size_caps_at_subset_size(self) -> None:
        self.assertEqual(_resolve_overfit_batch_size(None, configured_batch_size=16, subset_size=8), 8)

    def test_explicit_overfit_batch_size_overrides_config(self) -> None:
        self.assertEqual(_resolve_overfit_batch_size(4, configured_batch_size=2, subset_size=8), 4)


if __name__ == "__main__":
    unittest.main()