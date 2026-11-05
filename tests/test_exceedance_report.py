import unittest

import matplotlib

matplotlib.use("Agg")

import pandas as pd

import exceedance_report as er


def _daily():
    days = pd.date_range("2025-01-01", periods=40, freq="D")
    rows = []
    for i, day in enumerate(days):
        # A: 30 exceedances, 15 of them removed by the deduction -> daily LV met after deduction.
        meas_a = 60.0 if i < 30 else 20.0
        contrib_a = 30.0 if i < 15 else 0.0
        rows.append(("ES/A", day, meas_a, meas_a - contrib_a, contrib_a, contrib_a > 0, "UB"))
        # B: no exceedances, no dust.
        rows.append(("ES/B", day, 10.0, 10.0, 0.0, False, "RB"))
    return pd.DataFrame(rows, columns=["Samplingpoint", "day", "daily_mean", "corrected_pollutant",
                                       "Dust_contribution", "dust_flag", "station_class"])


class ExceedanceReportTest(unittest.TestCase):
    def setUp(self):
        self.stations = er.station_compliance(_daily(), daily_limit=45, annual_limit=40, allowed_exceedances=18)

    def test_station_compliance(self):
        a = self.stations.set_index("Samplingpoint").loc["ES/A"]
        self.assertEqual(a["exceedances_measured"], 30)
        self.assertEqual(a["exceedances_adjusted"], 15)
        self.assertEqual(a["exceedances_natural"], 15)
        self.assertTrue(a["daily_lv_exceeded_measured"])
        self.assertFalse(a["daily_lv_exceeded_adjusted"])
        self.assertEqual(a["daily_lv_status"], "met after deduction")
        # Annual mean 50 -> 38.75 with limit 40.
        self.assertEqual(a["annual_lv_status"], "met after deduction")
        self.assertEqual(a["overall_status"], "met after deduction")
        self.assertEqual(a["status_change"], "daily and annual LV")
        b = self.stations.set_index("Samplingpoint").loc["ES/B"]
        self.assertEqual((b["overall_status"], b["status_change"]), ("met", ""))

    def test_station_info_and_map(self):
        daily = _daily().drop(columns=["station_class"])
        daily["Samplingpoint"] = daily["Samplingpoint"] + " (UTC day)"
        info = pd.DataFrame({"station_id": ["ES/A", "ES/B"], "class": ["TH", "RB"],
                             "latitude": [40.0, 41.0], "longitude": [-3.0, -4.0]})
        daily = er.add_station_info(daily, info)
        stations = er.station_compliance(daily, daily_limit=45, annual_limit=40, allowed_exceedances=18)
        a = stations.set_index("Samplingpoint").loc["ES/A (UTC day)"]
        self.assertEqual((a["station_class"], a["Latitude"]), ("TH", 40.0))
        try:
            import cartopy  # noqa: F401
        except ImportError:
            self.skipTest("cartopy not installed")
        er.plot_compliance_map(stations)

    def test_summary_and_class(self):
        summary = er.compliance_summary(self.stations).set_index("indicator")
        row = summary.loc["Stations above the allowed number of exceedances (daily LV)"]
        self.assertEqual((row["measured"], row["after deduction"]), (1, 0))
        classes = er.class_summary(self.stations).set_index("station_class")
        self.assertEqual(classes.loc["UB", "share_natural_pct"], 50.0)

    def test_monthly_and_episodes(self):
        monthly = er.monthly_summary(_daily(), daily_limit=45).set_index("month")
        self.assertEqual(monthly.loc[1, "exceedances_measured"], 30)
        self.assertEqual(monthly.loc[1, "exceedances_natural"], 15)
        self.assertEqual(len(monthly), 12)
        episodes = er.episode_table(_daily(), daily_limit=45)
        self.assertEqual(len(episodes), 15)

    def test_plots_and_missing_columns(self):
        daily = _daily().drop(columns=["Dust_contribution", "dust_flag", "station_class"])
        stations = er.station_compliance(daily, daily_limit=45, annual_limit=40, allowed_exceedances=18)
        er.plot_station_exceedances(stations, allowed_exceedances=18)
        er.plot_monthly(er.monthly_summary(daily, daily_limit=45))
        er.plot_annual_means(stations, annual_limit=40)
        er.plot_contribution_distribution(daily)


if __name__ == "__main__":
    unittest.main()
