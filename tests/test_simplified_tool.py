"""Simplified tool (dust_tool + dust_plots) on a small synthetic run."""

import contextlib
import io
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import dust_plots as dp
import dust_tool as dt


def _write_metadata(folder):
    rows = [
        # station code, sampling point, area, type, lat, lon
        ("ES0001A", "SP_RB", "rural-regional", "background", 40.70, -3.70),
        ("ES0002A", "SP_UB", "urban", "background", 40.42, -3.70),
        ("ES0003A", "SP_TH", "urban", "traffic", 40.43, -3.69),
    ]
    meta = pd.DataFrame(rows, columns=["Air Quality Station EoI Code", "Sampling Point Id", "Air Quality Station Area",
                                       "Air Quality Station Type", "Latitude", "Longitude"])
    meta["Air Quality Station Name"] = meta["Sampling Point Id"]
    meta["Air Pollutant"] = "PM10"
    meta["Altitude"] = 650.0
    meta["Timezone"] = "UTC+01"
    meta.to_csv(os.path.join(folder, "DataExtract.csv"), index=False)


def _observations(year):
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    rng = np.random.default_rng(0)
    cams = np.where(np.arange(len(days)) % 20 == 0, 30.0, 1.0)  # a dust event every 20 days
    frames = []
    for sp, base, lat in (("ES/SP_RB", 15, 40.70), ("ES/SP_UB", 25, 40.42), ("ES/SP_TH", 35, 40.43)):
        values = base + rng.normal(0, 2, len(days)) + np.where(cams > 5, 30.0, 0.0)
        frames.append(pd.DataFrame({"Samplingpoint": sp, "day": days, "daily_mean": values, "cams_dust": cams,
                                    "Latitude": lat, "Longitude": -3.70}))
    return pd.concat(frames, ignore_index=True)


class SimplifiedToolTest(unittest.TestCase):
    def test_full_run(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            _write_metadata(tmp)
            settings = dt.Settings(year=2025, pollutant="PM10", countries=["ES"], temporal_flag="day",
                                   first_run=True, project_dir=tmp, use_google_drive=False)
            tool = dt.DustTool(settings)
            self.assertEqual((tool.daily_limit, tool.annual_limit), (45, 20))
            self.assertEqual(tool.plan, dt.RunPlan(download_eea=True, download_cams=True,
                                                   compute_cams_daily=True, load_parquet=False))

            tool.df = _observations(2025)
            tool.build_pairing()
            self.assertEqual(len(tool.pairing_session.table), 3)
            table = tool.compute_results(save_csv=True)
            self.assertTrue(os.path.exists(tool.deduction_parquet))
            self.assertEqual(len(table), 3)

            exc = dp.exceedance_map_table(table, tool.df_target_year, ["ES"], tool.daily_limit)
            self.assertTrue(set(exc["Daily_LV_compliance"]) <= {c for c, _ in dp.LV_MAP_CLASSES})
            annual = dp.annual_map_table(table, tool.annual_limit)
            self.assertIn("Annual_LV_compliance", annual)
            stations, report, _ = dp.station_compliance_tables(tool, 18)
            self.assertEqual(len(report), 3)

            # Later run: the saved Parquet is found, so the data steps are skipped.
            settings.first_run = False
            loaded = dt.DustTool(settings)
            self.assertTrue(loaded.plan.load_parquet)
            self.assertFalse(loaded.plan.download_eea or loaded.plan.download_cams)
            loaded.prepare_data()
            loaded.build_pairing()
            pd.testing.assert_frame_equal(loaded.compute_results(save_csv=False), table)

    def test_later_run_without_saved_results(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            tool = dt.DustTool(dt.Settings(first_run=False, project_dir=tmp, use_google_drive=False))
            self.assertEqual(tool.plan, dt.RunPlan(False, False, False, False))
            # An explicit switch wins over first_run.
            tool = dt.DustTool(dt.Settings(first_run=False, download_eea=True, project_dir=tmp,
                                           use_google_drive=False))
            self.assertTrue(tool.plan.download_eea)
            self.assertFalse(tool.plan.load_parquet)

    def test_bad_pairing_mode(self):
        with self.assertRaises(ValueError):
            dt.DustTool(dt.Settings(pairing_mode="other"))


if __name__ == "__main__":
    unittest.main()
