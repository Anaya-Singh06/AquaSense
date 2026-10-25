import unittest

import numpy as np

from tests._helpers import ROOT  # noqa: F401
from data.generate_synthetic_data import generate
from src.config import COMMON_COLUMNS, LABEL_COLUMN, TARGET
from src.spatial.idw import build_spatial_grid, haversine_km, idw_interpolate


class TestSynthetic(unittest.TestCase):
    def test_shape_schema_and_seed(self):
        df, st = generate()
        self.assertEqual(df["sensor_id"].nunique(), 10)
        self.assertEqual(df.groupby("sensor_id").size().unique().tolist(), [730])
        for c in [*COMMON_COLUMNS, LABEL_COLUMN]:
            self.assertIn(c, df.columns)
        self.assertTrue(df[TARGET].isna().any())             # missing values / dropouts present
        self.assertGreater(df[LABEL_COLUMN].sum(), 0)
        self.assertEqual(len(st), 10)
        df2, _ = generate()
        self.assertTrue(df.equals(df2))                       # fixed seed -> identical
        self.assertFalse(df.equals(generate(seed=7)[0]))

    def test_physical_ranges(self):
        df, _ = generate()
        self.assertGreaterEqual(df[TARGET].min(), 0); self.assertLess(df[TARGET].max(), 60)
        self.assertGreaterEqual(df["rainfall_mm"].min(), 0)
        self.assertTrue(df["soil_moisture"].dropna().between(0, 1).all())
        monsoon = df[df["date"].dt.month.isin([6, 7, 8, 9])]["rainfall_mm"].mean()
        dry = df[df["date"].dt.month.isin([1, 2, 3])]["rainfall_mm"].mean()
        self.assertGreater(monsoon, 3 * dry)


class TestIDW(unittest.TestCase):
    def test_exact_station_and_midpoint(self):
        lat, lon, val = np.array([10., 10.]), np.array([78., 79.]), np.array([10., 20.])
        v, near = idw_interpolate(lat, lon, val, np.array([10., 10.]), np.array([78., 78.5]), 2.0)
        self.assertAlmostEqual(v[0], 10.0); self.assertAlmostEqual(v[1], 15.0, places=1)
        self.assertLess(near[0], 1e-6)

    def test_bounds_and_power_effect(self):
        rng = np.random.default_rng(0)
        lat, lon, val = rng.uniform(10, 11, 6), rng.uniform(78, 79, 6), rng.uniform(5, 25, 6)
        q = (np.array([10.5]), np.array([78.5]))
        v2, _ = idw_interpolate(lat, lon, val, *q, power=2)
        self.assertTrue(val.min() <= v2[0] <= val.max())
        v6, _ = idw_interpolate(lat, lon, val, *q, power=6)
        d = haversine_km(q[0][:, None], q[1][:, None], lat[None], lon[None])[0]
        self.assertLess(abs(v6[0] - val[d.argmin()]), abs(v2[0] - val[d.argmin()]) + 1e-9)  # higher p -> closer to nearest

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            idw_interpolate(np.array([1.]), np.array([1.]), np.array([1.]), np.array([1.]), np.array([1.]), power=0)
        with self.assertRaises(ValueError):
            idw_interpolate(np.array([]), np.array([]), np.array([]), np.array([1.]), np.array([1.]))

    def test_geojson(self):
        g = build_spatial_grid(np.array([10., 10.5, 11.]), np.array([78., 78.5, 79.]), np.array([8., 15., 25.]), n=8, max_distance_km=None)
        self.assertEqual(g["type"], "FeatureCollection"); self.assertEqual(len(g["features"]), 64)
        self.assertEqual({f["properties"]["status"] for f in g["features"]} <= {"SAFE", "SEMI_CRITICAL", "CRITICAL"}, True)
        ring = g["features"][0]["geometry"]["coordinates"][0]
        self.assertEqual(ring[0], ring[-1])
        self.assertIn("Prototype", g["properties"]["disclaimer"])
        single = build_spatial_grid(np.array([10.]), np.array([78.]), np.array([12.]), n=4, max_distance_km=None)
        self.assertTrue(all(f["properties"]["depth_m_bgl"] == 12.0 for f in single["features"]))


if __name__ == "__main__":
    unittest.main()
