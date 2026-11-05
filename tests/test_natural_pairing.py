"""Pairing, window selection and natural-contribution deduction."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import natural_contribution as nc
import pairing as pr


def _stations():
    """ES stations around Madrid plus one French RB closer than any ES RB."""
    rows = [
        # id, name, area, type, lat, lon, alt
        ("ES/RB_NEAR_HIGH", "RB near but high", "rural-regional", "background", 40.45, -3.70, 1400.0),
        ("ES/RB_OK", "RB 30 km", "rural", "background", 40.70, -3.70, 700.0),
        ("ES/RB_FAR", "RB 300 km", "rural-remote", "background", 43.10, -3.70, 650.0),
        ("ES/UB1", "UB Madrid", "urban", "background", 40.42, -3.70, 650.0),
        ("ES/TH1", "Traffic Madrid", "urban", "traffic", 40.43, -3.69, 660.0),
        ("ES/UB_ISLAND", "UB island", "urban", "background", 28.10, -15.40, 20.0),
        ("FR/RB_X", "French RB", "rural", "background", 40.43, -3.68, 650.0),
    ]
    meta = pd.DataFrame(
        rows,
        columns=["sp", "Air Quality Station Name", "Air Quality Station Area",
                 "Air Quality Station Type", "Latitude", "Longitude", "Altitude"],
    )
    meta["Air Quality Station EoI Code"] = meta["sp"].str[:2] + "0001A"
    meta["Sampling Point Id"] = meta["sp"].str[3:]
    return pr.build_station_table(meta)


class ClassificationTest(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(pr.classify_station("rural-near_city", "background"), "RB")
        self.assertEqual(pr.classify_station("suburban", "background"), "UB")
        self.assertEqual(pr.classify_station("rural", "industrial"), "TH")
        self.assertEqual(pr.classify_station("urban", "traffic"), "TH")
        self.assertEqual(pr.classify_station(None, None), "unknown")


class DefaultPairingTest(unittest.TestCase):
    def setUp(self):
        self.stations = _stations()
        self.table = pr.default_pairing(self.stations).set_index("station_id")

    def test_nearest_rb_respects_altitude_and_country(self):
        row = self.table.loc["ES/UB1"]
        self.assertEqual(row["paired_rb_id"], "ES/RB_OK")
        self.assertEqual(row["method"], "option1")
        self.assertAlmostEqual(row["rb_distance_km"], 31.1, delta=1.0)
        self.assertEqual(row["rb_altitude_diff_m"], 50.0)

    def test_rb_is_its_own_reference(self):
        row = self.table.loc["ES/RB_OK"]
        self.assertEqual(row["paired_rb_id"], "ES/RB_OK")
        self.assertEqual(row["rb_distance_km"], 0.0)

    def test_th_gets_rb_and_ub(self):
        row = self.table.loc["ES/TH1"]
        self.assertEqual(row["paired_rb_id"], "ES/RB_OK")
        self.assertEqual(row["paired_ub_id"], "ES/UB1")

    def test_no_reference_within_distance(self):
        row = self.table.loc["ES/UB_ISLAND"]
        self.assertIsNone(row["paired_rb_id"])
        self.assertEqual(row["method"], "option2")

    def test_th_without_rb_uses_ub(self):
        limits = pr.PairingLimits(max_rb_distance_km=5)
        table = pr.default_pairing(self.stations, limits).set_index("station_id")
        self.assertEqual(table.loc["ES/TH1", "method"], "option2_ub_paired")
        self.assertEqual(table.loc["ES/UB1", "method"], "option2")

    def test_manual_edit_and_reset(self):
        default = pr.default_pairing(self.stations)
        edited, warnings = pr.set_manual_pair(default, self.stations, "ES/UB1", rb_id="ES/RB_FAR")
        row = edited.set_index("station_id").loc["ES/UB1"]
        self.assertEqual(row["source"], "manual")
        self.assertEqual(row["paired_rb_name"], "RB 300 km")
        self.assertGreater(row["rb_distance_km"], 250)
        self.assertTrue(any("km away" in w for w in warnings))
        forced, _ = pr.set_manual_pair(default, self.stations, "ES/UB1", rb_id=None)
        self.assertEqual(forced.set_index("station_id").loc["ES/UB1", "method"], "option2")
        with self.assertRaises(pr.PairingValidationError):
            pr.set_manual_pair(default, self.stations, "ES/UB1", rb_id="ES/TH1")
        back = pr.reset_station(edited, default, "ES/UB1").set_index("station_id")
        self.assertEqual(back.loc["ES/UB1", "paired_rb_id"], "ES/RB_OK")
        self.assertEqual(back.loc["ES/UB1", "source"], "default")


class PairingCsvRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.stations = _stations()
        self.default = pr.default_pairing(self.stations)

    def _roundtrip(self, edit):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, pr.USER_PAIRING_FILENAME)
            pr.export_pairing_csv(self.default, path)
            user = pr.read_pairing_csv(path)
            edit(user)
            user.to_csv(path, index=False)
            return pr.apply_user_pairing(pr.read_pairing_csv(path), self.stations, self.default)

    def test_unchanged_roundtrip_is_default(self):
        table, report = self._roundtrip(lambda user: None)
        self.assertTrue((table["source"] == "default").all())
        self.assertEqual(len(report.changes), 0)
        self.assertEqual(report.errors, [])

    def test_edit_is_marked_manual_and_recomputed(self):
        def edit(user):
            user.loc[user["station_id"] == "ES/UB1", "paired_rb_id"] = "ES/RB_FAR"
            user.loc[user["station_id"] == "ES/UB1", "rb_distance_km"] = "999"
            user.loc[user["station_id"] == "ES/UB1", "comment"] = "expert choice"

        table, report = self._roundtrip(edit)
        row = table.set_index("station_id").loc["ES/UB1"]
        self.assertEqual(row["source"], "manual")
        self.assertEqual(row["method"], "option1")
        self.assertEqual(row["comment"], "expert choice")
        self.assertLess(row["rb_distance_km"], 999)
        self.assertEqual(set(report.changes["station_id"]), {"ES/UB1"})
        self.assertTrue(any("km away" in w for w in report.warnings))

    def test_forced_method_none(self):
        def edit(user):
            user.loc[user["station_id"] == "ES/TH1", "method"] = "none"

        table, report = self._roundtrip(edit)
        self.assertEqual(table.set_index("station_id").loc["ES/TH1", "method"], "none")

    def test_validation_errors(self):
        def edit(user):
            user.loc[user["station_id"] == "ES/UB1", "paired_rb_id"] = "ES/DOES_NOT_EXIST"
            user.loc[user["station_id"] == "ES/TH1", "paired_ub_id"] = "ES/RB_OK"
            user.loc[user["station_id"] == "ES/RB_FAR", "paired_rb_id"] = "ES/UB1"
            user.loc[user["station_id"] == "ES/UB_ISLAND", "method"] = "option9"

        with self.assertRaises(pr.PairingValidationError) as ctx:
            self._roundtrip(edit)
        text = "\n".join(ctx.exception.errors)
        self.assertIn("unknown station id 'ES/DOES_NOT_EXIST'", text)
        self.assertIn("ES/RB_OK is a RB station, not UB", text)
        self.assertIn("ES/UB1 is a UB station, not RB", text)
        self.assertIn("option9", text)

    def test_non_strict_keeps_default_for_bad_rows(self):
        user = self.default.copy()
        user.loc[user["station_id"] == "ES/UB1", "paired_rb_id"] = "XX/NOPE"
        table, report = pr.apply_user_pairing(user, self.stations, self.default, strict=False)
        self.assertEqual(table.set_index("station_id").loc["ES/UB1", "paired_rb_id"], "ES/RB_OK")
        self.assertEqual(len(report.errors), 1)


class WindowSelectionTest(unittest.TestCase):
    def test_event_days_are_excluded_from_window(self):
        values = np.arange(10, 19, dtype=float)  # index i -> 10 + i
        events = np.zeros(9, dtype=bool)
        events[[4, 6]] = True
        pool = nc.reference_window(values, events, 4, half_window=3, use_episode=False)
        np.testing.assert_array_equal(pool, [11, 12, 13, 15, 17])

    def test_missing_values_and_target_dropped(self):
        values = np.array([10, np.nan, 12, 50, 14], dtype=float)
        events = np.zeros(5, dtype=bool)
        pool = nc.reference_window(values, events, 3, half_window=2, use_episode=False)
        np.testing.assert_array_equal(pool, [12, 14])

    def test_multi_day_episode_uses_days_around_the_episode(self):
        values = np.array([1, 2, 3, 4, 90, 95, 99, 5, 6, 7, 8], dtype=float)
        events = np.zeros(11, dtype=bool)
        events[[4, 5, 6]] = True
        for target in (4, 5, 6):
            pool = nc.reference_window(values, events, target, half_window=3, use_episode=True)
            np.testing.assert_array_equal(pool, [2, 3, 4, 5, 6, 7])
        ref, n = nc.reference_for_days(
            values, events, [4, 5, 6], half_window=3, percentile=50,
            min_valid=3, use_episode=True,
        )
        np.testing.assert_allclose(ref, [4.5, 4.5, 4.5])
        np.testing.assert_array_equal(n, [6, 6, 6])

    def test_insufficient_reference(self):
        values = np.array([np.nan, np.nan, 30, np.nan, np.nan], dtype=float)
        events = np.array([False, False, True, False, False])
        ref, n = nc.reference_for_days(
            values, events, [2], half_window=2, percentile=50, min_valid=1, use_episode=True,
        )
        self.assertTrue(np.isnan(ref[0]))
        self.assertEqual(n[0], 0)


def _series(station, values, dust_days, start="2024-06-01"):
    days = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "Samplingpoint": station,
            "day": days,
            "daily_mean": values,
            "cams_dust": [12.0 if i in dust_days else 1.0 for i in range(len(values))],
        }
    )


def _pairing(rows):
    table = pd.DataFrame(rows, columns=["station_id", "class", "method", "paired_rb_id", "paired_ub_id"])
    table["source"] = "default"
    return table


class DeductionTest(unittest.TestCase):
    params = nc.MethodParams(min_reference_days_option1=3, min_reference_days_option2=3)

    def test_option1_load_is_applied_to_paired_station(self):
        rb = _series("ES/RB", [10] * 15 + [40] + [10] * 15, dust_days={15})
        ub = _series("ES/UB", [30] * 15 + [90] + [30] * 15, dust_days={15})
        pairing = _pairing([
            ("ES/RB", "RB", "option1", "ES/RB", None),
            ("ES/UB", "UB", "option1", "ES/RB", None),
        ])
        out = nc.compute_natural_contribution(
            pd.concat([rb, ub], ignore_index=True), pairing, dust_threshold=5, params=self.params,
        )
        dust = out[out["cams_dust_flag"]].set_index("Samplingpoint")
        self.assertAlmostEqual(dust.loc["ES/RB", "natural_contribution"], 30.0)
        self.assertAlmostEqual(dust.loc["ES/UB", "natural_contribution"], 30.0)
        self.assertAlmostEqual(dust.loc["ES/UB", "adjusted_value"], 60.0)
        self.assertEqual(dust.loc["ES/UB", "reference_station_id"], "ES/RB")
        self.assertTrue(dust["confirmed_dust_day"].all())
        self.assertEqual(out.loc[~out["cams_dust_flag"], "natural_contribution"].sum(), 0)
        again = nc.compute_natural_contribution(out, pairing, dust_threshold=5, params=self.params)
        pd.testing.assert_series_equal(again["adjusted_value"], out["adjusted_value"])

    def test_th_station_uses_ub_load_not_its_own(self):
        ub = _series("ES/UB", [20, 20, 20, 50, 20, 20, 20], dust_days={3})
        th = _series("ES/TH", [60, 70, 80, 75, 90, 60, 65], dust_days={3})
        pairing = _pairing([
            ("ES/UB", "UB", "option2", None, None),
            ("ES/TH", "TH", "option2_ub_paired", None, "ES/UB"),
        ])
        out = nc.compute_natural_contribution(
            pd.concat([ub, th], ignore_index=True), pairing, dust_threshold=5, params=self.params,
        )
        th_day = out[(out["Samplingpoint"] == "ES/TH") & out["cams_dust_flag"]].iloc[0]
        # UB load = 50 - 20 = 30; TH self-calibration would give 75 - 70 = 5.
        self.assertAlmostEqual(th_day["natural_contribution"], 30.0)
        self.assertAlmostEqual(th_day["adjusted_value"], 45.0)
        self.assertEqual(th_day["reference_station_id"], "ES/UB")
        self.assertEqual(th_day["method"], "option2_ub_paired")

    def test_adjusted_value_never_below_zero_and_insufficient_flag(self):
        ub = _series("ES/UB", [5, 5, 5, 3, 5, 5, 5], dust_days={3})
        rb = _series("ES/RB", [np.nan] * 7, dust_days={3})
        th = _series("ES/TH", [1, 1, 1, 1, 1, 1, 1], dust_days={3})
        pairing = _pairing([
            ("ES/UB", "UB", "option2", None, None),
            ("ES/TH", "TH", "option1", "ES/RB", None),
            ("ES/RB", "RB", "option1", "ES/RB", None),
        ])
        out = nc.compute_natural_contribution(
            pd.concat([ub, rb, th], ignore_index=True), pairing, dust_threshold=5, params=self.params,
        )
        dust = out[out["cams_dust_flag"]].set_index("Samplingpoint")
        self.assertEqual(dust.loc["ES/UB", "natural_contribution"], 0)
        self.assertFalse(dust.loc["ES/UB", "confirmed_dust_day"])
        self.assertTrue(dust.loc["ES/TH", "insufficient_reference"])
        self.assertEqual(dust.loc["ES/TH", "adjusted_value"], 1)
        self.assertTrue((out["adjusted_value"].dropna() >= 0).all())

    def test_annual_mean_uses_every_dust_day(self):
        # Two dust days, neither above the 45 µg/m³ limit.
        values = [20] * 10 + [35] + [20] * 10 + [30] + [20] * 10
        ub = _series("ES/UB", values, dust_days={10, 21})
        pairing = _pairing([("ES/UB", "UB", "option2", None, None)])
        out = nc.compute_natural_contribution(
            ub, pairing, dust_threshold=5, params=self.params, daily_limit=45,
        )
        self.assertEqual(int(out["Exceedance"].sum()), 0)
        self.assertAlmostEqual(out["natural_contribution"].sum(), 15 + 10)
        summary = nc.station_indicators(out, daily_limit=45, annual_limit=20).iloc[0]
        self.assertAlmostEqual(summary["annual_mean_measured"], np.mean(values), places=4)
        self.assertAlmostEqual(summary["annual_mean_adjusted"], 20.0, places=4)
        self.assertEqual(summary["confirmed_dust_days"], 2)
        self.assertTrue(summary["annual_limit_exceeded_measured"])
        self.assertFalse(summary["annual_limit_exceeded_adjusted"])

    def test_exceedance_counts(self):
        values = [30] * 5 + [80] + [30] * 5 + [60] + [30] * 5
        ub = _series("ES/UB", values, dust_days={5})
        pairing = _pairing([("ES/UB", "UB", "option2", None, None)])
        out = nc.compute_natural_contribution(
            ub, pairing, dust_threshold=5, params=self.params, daily_limit=45,
        )
        summary = nc.station_indicators(out, daily_limit=45, annual_limit=40).iloc[0]
        self.assertEqual(summary["exceedances_measured"], 2)
        self.assertEqual(summary["exceedances_adjusted"], 1)
        self.assertEqual(summary["exceedances_natural"], 1)
        self.assertEqual(summary["Dust_exceedance_days"], 1)

    def test_limits(self):
        limits = nc.LimitValues()
        self.assertEqual(limits.daily("PM10"), 45)
        self.assertEqual(limits.daily("PM2.5"), 25)
        self.assertEqual(nc.LimitValues(pm10_daily=50).daily("PM10"), 50)


class ModelThresholdTest(unittest.TestCase):
    def test_lowest_threshold_with_consistent_contribution(self):
        import dust_threshold_sensitivity as sens

        # Low CAMS values (3-4) come with no PM increase; 8+ with a clear one.
        values, cams = [], []
        for block in range(12):
            values += [20, 20, 20, 20, 20, 20, 20]
            cams += [1, 1, 1, 1, 1, 1, 1]
            values += [20, 20, 20, 50, 20, 20, 20]
            cams += [1, 1, 1, 9, 1, 1, 1]
            values += [20, 20, 20, 19, 20, 20, 20]
            cams += [1, 1, 1, 4, 1, 1, 1]
        days = pd.date_range("2024-01-01", periods=len(values), freq="D")
        df = pd.DataFrame({"Samplingpoint": "ES/UB", "day": days, "daily_mean": values, "cams_dust": cams})
        pairing = _pairing([("ES/UB", "UB", "option2", None, None)])
        frames = sens.natural_threshold_sweep(df, pairing, [3, 5, 8])
        table = sens.contribution_consistency(frames)
        result = sens.lowest_consistent_threshold(table, min_share=0.9, min_days=5).iloc[0]
        self.assertEqual(result["country"], "ES")
        self.assertEqual(result["model_threshold_ug_m3"], 5.0)
        low = table[table["dust_threshold_ug_m3"] == 3].iloc[0]
        self.assertAlmostEqual(low["share_confirmed"], 0.5)


if __name__ == "__main__":
    unittest.main()
