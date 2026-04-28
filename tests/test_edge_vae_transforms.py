from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

from edge_vae.transforms import (
    EDGE_DEPTH_NORMALIZATION_SCALE,
    compute_df_void_map,
    decode_df_tensor_to_edge_depth,
    decode_raw_tensor_to_edge_depth,
    encode_edge_depth_to_df_tensor,
    encode_edge_depth_to_raw_tensor,
)


class TestEdgeVaeTransforms(unittest.TestCase):
    def test_compute_df_void_map_respects_formula_endpoints(self) -> None:
        valid_mask = np.array([[True, False]], dtype=bool)
        diag = math.sqrt(5.0)
        with mock.patch("edge_vae.transforms.distance_transform_edt", return_value=np.array([[0.0, diag]], dtype=np.float32)):
            df = compute_df_void_map(valid_mask, beta=30.0)
        self.assertAlmostEqual(float(df[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(df[0, 1]), -1.0, places=6)

    def test_df_encoding_keeps_valid_nonnegative_and_void_nonpositive(self) -> None:
        edge_depth = np.array(
            [
                [[0.0, 0.5], [1.0, 0.0]],
                [[0.25, 0.0], [0.0, 0.75]],
                [[0.0, 0.0], [0.5, 0.25]],
            ],
            dtype=np.float32,
        )
        encoded = encode_edge_depth_to_df_tensor(edge_depth, beta=30.0)
        valid_mask = edge_depth > 1.0e-8
        self.assertTrue(np.all(encoded[valid_mask] >= 0.0))
        self.assertTrue(np.all(encoded[valid_mask] <= 1.0))
        self.assertTrue(np.allclose(encoded[valid_mask], edge_depth[valid_mask] * EDGE_DEPTH_NORMALIZATION_SCALE))
        self.assertTrue(np.all(encoded[~valid_mask] <= 0.0))
        self.assertTrue(np.all(encoded[~valid_mask] >= -1.0))

    def test_raw_encoding_keeps_valid_within_raw_scale_and_void_zero(self) -> None:
        raw_scale = 1.0 / math.sqrt(3.0)
        edge_depth = np.array(
            [
                [[0.0, 0.5], [1.0, 0.0]],
                [[0.25, 0.0], [0.0, 0.75]],
                [[0.0, 0.0], [0.5, 0.25]],
            ],
            dtype=np.float32,
        )
        encoded = encode_edge_depth_to_raw_tensor(edge_depth, raw_scale=raw_scale)
        valid_mask = edge_depth > 1.0e-8
        self.assertTrue(np.all(encoded[valid_mask] >= 0.0))
        self.assertTrue(np.allclose(encoded[valid_mask], edge_depth[valid_mask] * raw_scale))
        self.assertTrue(np.all(encoded[~valid_mask] == 0.0))

    def test_no_valid_hit_channel_uses_expected_fill_values(self) -> None:
        edge_depth = np.zeros((3, 4, 4), dtype=np.float32)
        edge_depth[1, 1, 1] = 0.5
        encoded_df = encode_edge_depth_to_df_tensor(edge_depth, beta=30.0)
        encoded_raw = encode_edge_depth_to_raw_tensor(edge_depth)
        self.assertTrue(np.all(encoded_df[0] == -1.0))
        self.assertTrue(np.all(encoded_df[2] == -1.0))
        self.assertTrue(np.all(encoded_raw[0] == 0.0))
        self.assertTrue(np.all(encoded_raw[2] == 0.0))

    def test_decode_threshold_behavior_for_df_and_raw(self) -> None:
        encoded = np.array(
            [
                [[-0.5, 0.01], [0.02, 0.03]],
                [[0.0, 0.5], [0.019, 0.021]],
                [[-1.0, -0.1], [0.5, 1.0]],
            ],
            dtype=np.float32,
        )
        decoded_df = decode_df_tensor_to_edge_depth(encoded, valid_threshold=0.02)
        decoded_raw = decode_raw_tensor_to_edge_depth(encoded, raw_scale=0.5, valid_threshold=0.02)
        self.assertEqual(float(decoded_df[0, 0, 1]), 0.0)
        self.assertEqual(float(decoded_df[0, 1, 0]), 0.0)
        self.assertGreater(float(decoded_df[0, 1, 1]), 0.0)
        self.assertEqual(float(decoded_raw[1, 1, 0]), 0.0)
        self.assertGreater(float(decoded_raw[1, 1, 1]), 0.0)

    def test_df_encode_decode_uses_same_scale(self) -> None:
        edge_depth = np.array(
            [
                [[0.0, 0.5], [1.0, 0.0]],
                [[0.25, 0.0], [0.0, 0.75]],
                [[0.0, 0.0], [0.5, 0.25]],
            ],
            dtype=np.float32,
        )
        encoded = encode_edge_depth_to_df_tensor(edge_depth, beta=30.0)
        decoded = decode_df_tensor_to_edge_depth(encoded, valid_threshold=0.0)
        valid_mask = edge_depth > 1.0e-8
        self.assertTrue(np.allclose(decoded[valid_mask], edge_depth[valid_mask], atol=1.0e-6))

    def test_raw_encode_decode_uses_same_scale(self) -> None:
        raw_scale = 1.0 / math.sqrt(3.0)
        edge_depth = np.array(
            [
                [[0.0, 0.5], [1.0, 0.0]],
                [[0.25, 0.0], [0.0, 0.75]],
                [[0.0, 0.0], [0.5, 0.25]],
            ],
            dtype=np.float32,
        )
        encoded = encode_edge_depth_to_raw_tensor(edge_depth, raw_scale=raw_scale)
        decoded = decode_raw_tensor_to_edge_depth(encoded, raw_scale=raw_scale, valid_threshold=0.0)
        self.assertTrue(np.allclose(decoded, edge_depth, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()