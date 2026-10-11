"""Regression checks for physically non-negative mining-area lower estimates.

Run with: python -m unittest discover -s 3-mining-area-simulation -p test_nonnegative_area.py
Only NumPy and SciPy are required; no full raster inputs are loaded.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]


def notebook_function(relative_path, name, namespace):
    notebook = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "".join(cell["source"]).startswith(f"def {name}(")
    )
    exec(compile(source, str(relative_path), "exec"), namespace)
    return namespace[name]


class Raster:
    def __init__(self, values, crs="test"):
        self.values = values
        self.shape = values.shape
        self.crs = crs
        self.transform = (50000, 0, 0, 0, -50000, 0)

    def read(self, band):
        return self.values


class RasterIO:
    def __init__(self, allocation):
        self.allocation = allocation
        self.written = None

    def open(self, path, mode="r", **kwargs):
        if mode == "w":
            return self
        return Raster(self.allocation)

    def write(self, values, band):
        self.written = values.copy()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class NonnegativeAreaTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {"np": np, "t": t, "Path": Path, "RESULTS": Path("unused")}
        self.interval = notebook_function(
            "3-mining-area-simulation/6-estimate-mining-area.ipynb",
            "predict_interval", self.namespace,
        )

    def test_lower_floor_zero_demand_and_unchanged_upper(self):
        values = np.array([0, 0.0001, 1], dtype=np.float32)
        lower, upper = self.interval(values, (1, 1), np.diag([0, 0.01]), 100)
        self.assertEqual(lower[0], 0)
        self.assertEqual(upper[0], 0)
        self.assertEqual(lower[1], 0)  # Previously negative for a small extraction amount.
        self.assertEqual(lower[2], 1)
        expected_upper = values[1] + t.ppf(0.975, 100) * np.sqrt(0.01 * (values[1] * np.log(values[1])) ** 2)
        self.assertAlmostEqual(float(upper[1]), float(expected_upper), places=9)
        self.assertEqual(upper[2], 1)
        self.assertEqual(lower.dtype, np.float32)
        self.assertTrue(np.all(lower <= upper))

    def test_positive_lower_estimates_are_preserved(self):
        values = np.array([0.1, 1, 10], dtype=np.float32)
        lower, upper = self.interval(values, (1, 1), np.zeros((2, 2)), 100)
        np.testing.assert_array_equal(lower, values)
        np.testing.assert_array_equal(upper, values)

    def test_lower_floor_precedes_subcell_sum(self):
        io = RasterIO(np.array([[1.0001]], dtype=np.float64))
        self.namespace["rio"] = io
        estimate = notebook_function(
            "3-mining-area-simulation/6-estimate-mining-area.ipynb",
            "estimate_mining_area", self.namespace,
        )
        cap = np.zeros((50, 50), dtype=np.float32)
        cap[0, :2] = [1, 0.0001]
        fit = (1, 1, (1, 1), np.diag([0, 0.01]), 100)
        with tempfile.TemporaryDirectory() as folder:
            estimate("BAU", 0, area_level="L", suit_1km=np.ones_like(cap),
                     cap_1km=cap, fit=fit, output_dir=folder)
        # The small sub-cell contributes zero. Flooring only the final cell sum
        # would still let its negative lower estimate reduce this positive area.
        np.testing.assert_array_equal(io.written, np.array([[0.01]], dtype=np.float32))

    def test_invalid_area_is_rejected_before_writing(self):
        io = RasterIO(np.array([[1]], dtype=np.float32))
        self.namespace["rio"] = io
        estimate = notebook_function(
            "3-mining-area-simulation/6-estimate-mining-area.ipynb",
            "estimate_mining_area", self.namespace,
        )
        cap = np.ones((50, 50), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "finite, non-negative"):
            estimate("BAU", 0, suit_1km=cap, cap_1km=cap,
                     fit=(-1, 1, (-1, 1), np.zeros((2, 2)), 100))
        self.assertIsNone(io.written)

    def test_risk_rejects_stale_negative_footprints(self):
        self.namespace["rio"] = RasterIO(np.array([[-0.00001]], dtype=np.float32))
        assess = notebook_function(
            "4-biodiversity-risk-assessment/6-biodiversity-risk-assessment.ipynb",
            "assess_biodiversity_risk", self.namespace,
        )
        with self.assertRaisesRegex(ValueError, "regenerate"):
            assess("BAU", 0, area_level="L")


if __name__ == "__main__":
    unittest.main()
