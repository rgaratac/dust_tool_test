"""User-supplied observations and stations in the EEA layout."""

import os
import tempfile
import unittest

import pandas as pd

import cams_data_to_eea_daily as cams_daily
import pairing as pr


class UserObservationsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        self.obs_path, self.st_path = cams_daily.write_user_templates(self.folder, "day", "PM10")
        self.stations = cams_daily.load_user_stations(self.st_path, "PM10")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, frame):
        path = os.path.join(self.folder, name)
        frame.to_csv(path, index=False)
        return path

    def _obs(self):
        return pd.read_csv(self.obs_path, dtype=str)

    def test_deduction_parquet_name_and_legacy_file(self):
        new = cams_daily.deduction_parquet_path(self.folder, "PM10", "E2a", "hour", 2025)
        self.assertTrue(new.endswith("CAMS_dust_PM10_deduction_E2a_hour_2025.parquet"))
        self.assertEqual(cams_daily.find_deduction_parquet(self.folder, "PM10", "E2a", "hour", 2025), new)

        legacy = cams_daily.deduction_parquet_path(self.folder, "PM10", "E2a", "hour", 2025, 6)
        open(legacy, "wb").close()
        self.assertEqual(cams_daily.find_deduction_parquet(self.folder, "PM10", "E2a", "hour", 2025), legacy)

        open(new, "wb").close()
        self.assertEqual(cams_daily.find_deduction_parquet(self.folder, "PM10", "E2a", "hour", 2025), new)

    def test_templates_load_as_daily_table(self):
        daily = cams_daily.load_user_observations(self.obs_path, self.stations, "day", "PM10")
        self.assertEqual(list(daily.columns), ["Pollutant", "day", "Samplingpoint", "daily_mean", "unit"])
        self.assertEqual(len(daily), 6)
        self.assertEqual(daily["day"].min(), pd.Timestamp("2025-01-01"))

        meta = cams_daily.combine_station_metadata(None, self.stations)
        table = pr.build_station_table(meta, pollutant="PM10", countries=["ES"]).set_index("station_id")
        self.assertEqual(table.loc["ES/SP_MY_STATION_1", "class"], "RB")
        self.assertEqual(table.loc["ES/SP_MY_STATION_2", "class"], "TH")

    def test_hourly_uses_eea_utc_plus_one_and_complete_days(self):
        hours = pd.date_range("2025-03-01 01:00", periods=30, freq="h")
        obs = pd.DataFrame(
            {
                "Samplingpoint": "ES/SP_MY_STATION_1", "Pollutant": "5",
                "Start": hours.strftime("%Y-%m-%d %H:%M:%S"),
                "End": (hours + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "Value": 10.0, "Unit": "ug.m-3", "AggType": "hour",
            }
        )
        daily = cams_daily.load_user_observations(self._write("obs.csv", obs), self.stations, "hour", "PM10")
        # 01:00 local = 00:00 UTC, so 2025-03-01 UTC has all 24 hours.
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily["day"].iloc[0], pd.Timestamp("2025-03-01", tz="UTC"))

    def test_missing_column_and_empty_value(self):
        with self.assertRaises(KeyError):
            cams_daily.load_user_observations(
                self._write("cols.csv", self._obs().drop(columns="Unit")), self.stations
            )
        obs = self._obs()
        obs.loc[0, "Unit"] = None
        with self.assertRaises(ValueError):
            cams_daily.load_user_observations(self._write("empty.csv", obs), self.stations)

    def test_unknown_station_and_wrong_unit(self):
        obs = self._obs()
        obs.loc[0, "Samplingpoint"] = "ES/NOT_IN_STATIONS"
        with self.assertRaises(ValueError):
            cams_daily.load_user_observations(self._write("unknown.csv", obs), self.stations)
        obs = self._obs()
        obs.loc[0, "Unit"] = "mg.m-3"
        with self.assertRaises(ValueError):
            cams_daily.load_user_observations(self._write("unit.csv", obs), self.stations)

    def test_optional_eea_flags_are_applied(self):
        obs = self._obs()
        daily = cams_daily.load_user_observations(self._write("plain.csv", obs), self.stations, "day", "PM10")
        self.assertEqual(len(daily), 6)
        obs["Validity"] = "1"
        obs["Verification"] = "1"
        obs.loc[0, "Validity"] = "-1"
        obs.loc[1, "Verification"] = "3"
        obs.loc[2, "Value"] = "-2"
        daily = cams_daily.load_user_observations(self._write("obs.csv", obs), self.stations, "day", "PM10")
        self.assertNotIn("ES/SP_MY_STATION_1", set(daily["Samplingpoint"]))
        obs.loc[3, "Validity"] = None
        with self.assertRaises(ValueError):
            cams_daily.load_user_observations(self._write("empty.csv", obs), self.stations, "day", "PM10")

    def test_station_checks(self):
        st = pd.read_csv(self.st_path, dtype=str)
        bad = st.copy()
        bad.loc[0, "Air Quality Station Type"] = "roadside"
        with self.assertRaises(ValueError):
            cams_daily.load_user_stations(self._write("type.csv", bad), "PM10")
        bad = st.copy()
        bad.loc[0, "Timezone"] = "CET"
        with self.assertRaises(ValueError):
            cams_daily.load_user_stations(self._write("tz.csv", bad), "PM10")
        with self.assertRaises(KeyError):
            cams_daily.load_user_stations(self._write("alt.csv", st.drop(columns="Altitude")), "PM10")

    def test_user_stations_override_dataextract(self):
        extract = pd.DataFrame(
            {
                "Air Quality Station EoI Code": ["ES9001A"],
                "Sampling Point Id": ["SP_MY_STATION_1"],
                "Latitude": [0.0],
                "Longitude": [0.0],
            }
        )
        meta = cams_daily.combine_station_metadata(self._write("DataExtract.csv", extract), self.stations)
        table = pr.build_station_table(meta).set_index("station_id")
        self.assertAlmostEqual(table.loc["ES/SP_MY_STATION_1", "latitude"], 40.70)


if __name__ == "__main__":
    unittest.main()
