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

    def test_local_day_follows_the_station_timezone(self):
        # 15 June 00:00-23:00 CET is the local day of a UTC+01 station; a UTC+02 station
        # needs 14 June 23:00 - 15 June 22:00 CET, a UTC station 15 June 01:00 - 16 June 00:00 CET.
        frame = pd.concat([
            _hourly_rows("ES/MAD", "2025-06-15 00:00", 24, value=10.0),
            _hourly_rows("GR/ATH", "2025-06-14 23:00", 24, value=20.0),
            _hourly_rows("PT/LIS", "2025-06-15 01:00", 24, value=30.0),
        ], ignore_index=True)
        offsets = pd.Series({"ES/MAD": 1, "GR/ATH": 2, "PT/LIS": 0})
        with tempfile.TemporaryDirectory() as folder:
            frame.to_parquet(os.path.join(folder, "obs.parquet"), index=False)
            daily = cams.load_and_filter_eea_hourly(folder, day_definition="local", tz_offsets=offsets)
            utc_daily = cams.load_and_filter_eea_hourly(folder)
        self.assertEqual(len(daily), 3)
        self.assertTrue((daily["day"] == pd.Timestamp("2025-06-15")).all())
        self.assertEqual(dict(zip(daily["Samplingpoint"].astype(str), daily["daily_mean"])),
                         {"ES/MAD": 10.0, "GR/ATH": 20.0, "PT/LIS": 30.0})
        # On UTC days only the UTC station has a complete 15 June.
        self.assertEqual(utc_daily["Samplingpoint"].astype(str).tolist(), ["PT/LIS"])

    def test_local_day_drops_stations_without_timezone(self):
        with tempfile.TemporaryDirectory() as folder:
            _hourly_rows("XX/NOTZ", "2025-06-15 00:00", 24).to_parquet(os.path.join(folder, "obs.parquet"))
            daily = cams.load_and_filter_eea_hourly(folder, day_definition="local", tz_offsets=pd.Series({"ES/A": 1}))
        self.assertTrue(daily.empty)


class LocalDayHelpersTest(unittest.TestCase):
    def test_hourly_request_reaches_into_the_next_year(self):
        self.assertTrue(cams.eea_request_body(["ES"], "PM10", "E1a", 2025, "hour")["dateTimeEnd"].startswith("2026-01-01"))
        self.assertTrue(cams.eea_request_body(["ES"], "PM10", "E1a", 2025, "day")["dateTimeEnd"].startswith("2025-12-31"))

    def test_product_tag_and_labels(self):
        self.assertEqual(cams.product_tag("hour", "local"), "hour_local")
        self.assertEqual(cams.product_tag("hour", "utc"), "hour")
        self.assertEqual(cams.product_tag("day", "local"), "day")
        frame = pd.DataFrame({"Samplingpoint": ["ES/A"], "day": [pd.Timestamp("2025-06-15")]})
        combined = cams.combine_hourly_and_daily(frame, frame.assign(Samplingpoint="ES/B"), hourly_day="local")
        self.assertEqual(combined["Samplingpoint"].tolist(),
                         ["ES/A (hourly, station local day)", "ES/B (station local day)"])
        import exceedance_report as er
        self.assertEqual(er.base_station_id("ES/A (hourly, station local day)"), "ES/A")

    def test_station_tz_offsets(self):
        meta = pd.DataFrame({"Air Quality Station EoI Code": ["ES0001A", "GR0002A"],
                             "Sampling Point Id": ["SP_1", "SP_2"], "Timezone": ["UTC+01", "UTC+02"]})
        self.assertEqual(cams.station_tz_offsets(meta).to_dict(), {"ES/SP_1": 1, "GR/SP_2": 2})


class PairingForProductTest(unittest.TestCase):
    def _table(self, rows):
        import pairing as pr
        frame = pd.DataFrame(rows)
        for col in pr.PAIRING_COLUMNS:
            if col not in frame.columns:
                frame[col] = None
        return frame[pr.PAIRING_COLUMNS]

    def test_reference_without_data_falls_back_to_the_product_default(self):
        import pairing as pr
        edited = self._table([
            {"station_id": "AT/UB_DAY", "class": "UB", "paired_rb_id": "AT/RB_HOUR", "method": "option1", "source": "manual"},
            {"station_id": "AT/UB_DAY2", "class": "UB", "paired_rb_id": "AT/RB_DAY", "method": "option1", "source": "default"},
            {"station_id": "AT/RB_DAY", "class": "RB", "paired_rb_id": "AT/RB_DAY", "method": "option1", "source": "default"},
            {"station_id": "AT/RB_HOUR", "class": "RB", "paired_rb_id": "AT/RB_HOUR", "method": "option1", "source": "default"},
        ])
        day_default = self._table([
            {"station_id": "AT/UB_DAY", "class": "UB", "paired_rb_id": "AT/RB_DAY", "method": "option1", "source": "default"},
            {"station_id": "AT/UB_DAY2", "class": "UB", "paired_rb_id": "AT/RB_DAY", "method": "option1", "source": "default"},
            {"station_id": "AT/RB_DAY", "class": "RB", "paired_rb_id": "AT/RB_DAY", "method": "option1", "source": "default"},
        ])
        table, notes = pr.pairing_for_product(edited, day_default, ["AT/UB_DAY", "AT/UB_DAY2", "AT/RB_DAY"])
        refs = dict(zip(table["station_id"], table["paired_rb_id"]))
        self.assertEqual(refs, {"AT/UB_DAY": "AT/RB_DAY", "AT/UB_DAY2": "AT/RB_DAY", "AT/RB_DAY": "AT/RB_DAY"})
        self.assertEqual(len(notes), 1)
        self.assertIn("AT/RB_HOUR", notes[0])


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
