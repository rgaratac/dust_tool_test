"""Hourly (UTC-day) path and the 'both' option. No network, no CAMS download."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

import cams_data_to_eea_daily as cams


class ResolutionsTest(unittest.TestCase):
    def test_both_runs_hour_then_day(self):
        self.assertEqual(cams.resolutions_for("both"), ["hour", "day"])
        self.assertEqual(cams.resolutions_for("day"), ["day"])
        with self.assertRaises(ValueError):
            cams.resolutions_for("month")

    def test_run_flags_for_both(self):
        flags = cams.resolve_run_flags(
            download_eea=False, download_cams=False, compute_cams_daily=False,
            load_parquet=True, eea_temporal_flag="both",
        )
        self.assertEqual(flags["resolutions"], ["hour", "day"])
        self.assertTrue(flags["local_day"])
        self.assertTrue(flags["utc_day"])
        self.assertTrue(flags["load_parquet"])


def _hourly_rows(station, start_cet, n_hours, value=10.0, **extra):
    stamps = pd.date_range(start_cet, periods=n_hours, freq="h")
    rows = pd.DataFrame({
        "Samplingpoint": station, "Pollutant": "PM10", "Start": stamps, "Value": value,
        "Unit": "ug.m-3", "AggType": "hour", "Validity": 1, "Verification": 1,
    })
    for col, val in extra.items():
        rows[col] = val
    return rows


class LoadEeaHourlyTest(unittest.TestCase):
    def _load(self, frame):
        with tempfile.TemporaryDirectory() as folder:
            frame.to_parquet(os.path.join(folder, "obs.parquet"), index=False)
            return cams.load_and_filter_eea_hourly(folder)

    def test_cet_stamps_become_one_full_utc_day(self):
        # 01:00 CET on 2 January is 00:00 UTC: 24 hours from there fill the UTC day of 2 January.
        daily = self._load(_hourly_rows("ES/A", "2025-01-02 01:00", 24, value=12.0))
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily["day"].iloc[0], pd.Timestamp("2025-01-02", tz="UTC"))
        self.assertAlmostEqual(float(daily["daily_mean"].iloc[0]), 12.0, places=5)

    def test_incomplete_and_invalid_hours_are_dropped(self):
        frame = pd.concat([
            _hourly_rows("ES/A", "2025-01-02 01:00", 24),
            _hourly_rows("ES/B", "2025-01-02 01:00", 23),                  # 23 hours: incomplete day
            _hourly_rows("ES/C", "2025-01-02 01:00", 24, Validity=-1),     # not valid
        ], ignore_index=True)
        daily = self._load(frame)
        self.assertEqual(daily["Samplingpoint"].astype(str).tolist(), ["ES/A"])

    def test_dispatch_by_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            _hourly_rows("ES/A", "2025-01-02 01:00", 24).to_parquet(os.path.join(folder, "obs.parquet"))
            self.assertEqual(len(cams.load_eea_observations(folder, "hour")), 1)
            with self.assertRaises(ValueError):
                cams.load_eea_observations(folder, "both")


class CamsUtcDailyTest(unittest.TestCase):
    def _grid(self):
        times = pd.date_range("2025-01-01", periods=3, freq="D")
        lat = np.array([40.0, 41.0])
        lon = np.array([-4.0, -3.0])
        dust = np.zeros((3, 2, 2), dtype="float32")
        for t in range(3):
            dust[t] = t + 1  # same value everywhere, different per day
        dust[:, 1, 1] += 10  # one corner differs, to check the interpolation
        return xr.Dataset({"dust": (("time", "lat", "lon"), dust)}, coords={"time": times, "lat": lat, "lon": lon})

    def test_values_per_station_and_day_and_domain_filter(self):
        df = pd.DataFrame({
            "Samplingpoint": ["ES/A", "ES/A", "ES/B", "XX/OUT"],
            "day": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-02", "2025-01-02"], utc=True),
            "Latitude": [40.0, 40.0, 40.5, 60.0],
            "Longitude": [-4.0, -4.0, -3.5, 20.0],
        })
        out = cams.add_cams_utc_daily_dust(self._grid(), df)
        self.assertNotIn("XX/OUT", out["Samplingpoint"].tolist())
        a = out[out["Samplingpoint"] == "ES/A"]["cams_dust"].tolist()
        self.assertEqual(a, [1.0, 3.0])
        # Centre of the cell: mean of the four corners, 2 + 10/4.
        self.assertAlmostEqual(float(out[out["Samplingpoint"] == "ES/B"]["cams_dust"].iloc[0]), 4.5, places=4)

    def test_missing_daily_file_returns_none(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(cams.cams_utc_daily_grid(folder, folder, 2025, compute=False))


if __name__ == "__main__":
    unittest.main()
