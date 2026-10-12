"""Conservation, overflow and physical-area checks without global raster inputs."""
import unittest
import numpy as np
from area_capacity import area_km2, extraction_limit, assign_extraction


class AreaCapacityTests(unittest.TestCase):
    def setUp(self):
        self.popt = (9.420545783094159, 0.5477038988759387)
        self.pcov = np.diag([0.04, 0.00001])

    def test_spill_preserves_volume_and_caps_all_area_cases(self):
        for level in ('L', 'M', 'H'):
            limit = extraction_limit(self.popt, self.pcov, 100, level)
            assigned = assign_extraction(116.82, np.minimum([116.82, 116.82], limit))
            self.assertAlmostEqual(assigned.sum(), 116.82, places=12)
            self.assertEqual(np.count_nonzero(assigned), 2)
            self.assertLessEqual(area_km2(assigned, self.popt, self.pcov, 100, level).max(), 1+1e-10)

    def test_zero_exact_capacity_and_infeasible_volume(self):
        np.testing.assert_array_equal(assign_extraction(0, [3, 5]), [0, 0])
        np.testing.assert_array_equal(assign_extraction(8, [3, 5]), [3, 5])
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            assign_extraction(9, [3, 5])
        with self.assertRaises(ValueError):
            assign_extraction(-1, [3, 5])

    def test_area_case_changes_capacity_before_allocation(self):
        limits = [extraction_limit(self.popt, self.pcov, 100, level) for level in ('L', 'M', 'H')]
        self.assertGreater(limits[0], limits[1])
        self.assertGreater(limits[1], limits[2])
        for level, limit in zip(('L', 'M', 'H'), limits):
            self.assertAlmostEqual(float(area_km2(limit, self.popt, self.pcov, 100, level)), 1., places=10)


if __name__ == '__main__':
    unittest.main()
