from unittest import TestCase

import rtctools.data.rtc as rtc
import rtctools.data.pi as pi

import numpy as np
import os
import datetime
import copy

from data_path import data_path


class TestPI(TestCase):

    def setUp(self):
        self.data_config = rtc.DataConfig(data_path())

    def test_parameter_config(self):
        parameter_config = pi.ParameterConfig(
            data_path(), 'rtcParameterConfig')
        self.assertEquals(parameter_config.get("group", "parameter"), 1.0)
        self.assertEquals(parameter_config.get(
            "group", "parameter", location_id="Location2"), 2.0)
        parameter_config.set("group", "parameter", 3.0,
                             location_id="Location2")
        self.assertEquals(parameter_config.get("group", "parameter"), 1.0)
        self.assertEquals(parameter_config.get(
            "group", "parameter", location_id="Location2"), 3.0)
        parameter_config.write()
        parameter_config = pi.ParameterConfig(
            data_path(), 'rtcParameterConfig')
        self.assertEquals(parameter_config.get("group", "parameter"), 1.0)
        self.assertEquals(parameter_config.get(
            "group", "parameter", location_id="Location2"), 3.0)
        parameter_config.set("group", "parameter", 2.0,
                             location_id="Location2")
        parameter_config.write()

    def test_timeseries(self):
        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        self.assertEquals(timeseries.get('S')[2], 2.0)
        self.assertTrue(np.isnan(timeseries.get('S')[3]))
        timeseries.write()

        # Ensure that NaNs remain NaNs after writing timeseries to file.
        self.assertTrue(np.isnan(timeseries.get('S')[3]))

        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        self.assertEquals(timeseries.get('S')[2], 2.0)

        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=True)
        self.assertEquals(timeseries.get('S')[2], 2.0)
        timeseries.write()

        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=True)
        self.assertEquals(timeseries.get('S')[2], 2.0)

        timeseries.set('S', timeseries.get('S'))

    def test_ensemble(self):
        timeseries = pi.Timeseries(self.data_config, data_path(
        ), "timeseries_import_ensemble", binary=False)
        self.assertEquals(timeseries.get('S', 0)[2], 2.0)
        self.assertEquals(timeseries.get('S', 1)[2], 3.0)

    def test_extend_timeseries(self):
        # Append item
        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        orig = np.array(timeseries.get('S', ensemble_member=0), copy=True)
        timeseries.resize(timeseries.start_datetime,
                          timeseries.end_datetime + timeseries.dt)
        timeseries.get('S', ensemble_member=0)[-1] = 12345.0
        timeseries.write()

        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        self.assertEquals(timeseries.get('S')[-1], 12345.0)
        self.assertEquals(len(timeseries.get('S')), len(orig) + 1)
        timeseries.set('S', orig, ensemble_member=0)
        timeseries.write()

        # Change unit
        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        timeseries._set_unit('S', unit='kcfs', ensemble_member=0)
        timeseries.write()

        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        self.assertEquals(timeseries._get_unit('S', ensemble_member=0), 'kcfs')
        timeseries._set_unit('S', unit='m3/s', ensemble_member=0)
        timeseries.write()

        # Remove last item
        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        orig = np.array(timeseries.get('S', ensemble_member=0), copy=True)
        timeseries.resize(timeseries.start_datetime,
                          timeseries.end_datetime - timeseries.dt)
        timeseries.write()

        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        self.assertEquals(timeseries.get('S')[0], orig[0])
        self.assertEquals(len(timeseries.get('S')), len(orig) - 1)
        timeseries.set('S', orig, ensemble_member=0)
        timeseries.write()

        # Prepend item
        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        orig = np.array(timeseries.get('S', ensemble_member=0), copy=True)
        timeseries.resize(timeseries.start_datetime -
                          timeseries.dt, timeseries.end_datetime)
        tmp = timeseries.get('S', ensemble_member=0)
        tmp[0] = 12345.0
        timeseries.set('S', tmp, ensemble_member=0)
        timeseries.write()

        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        self.assertEquals(timeseries.get('S')[0], 12345.0)
        self.assertEquals(len(timeseries.get('S')), len(orig) + 1)
        timeseries.set('S', orig, ensemble_member=0)
        timeseries.write()

        # Remove first item
        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        orig = np.array(timeseries.get('S', ensemble_member=0), copy=True)
        timeseries.resize(timeseries.start_datetime +
                          timeseries.dt, timeseries.end_datetime)
        timeseries.write()

        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import", binary=False)
        self.assertEquals(timeseries.get('S')[0], orig[1])
        self.assertEquals(len(timeseries.get('S')), len(orig) - 1)
        timeseries.set('S', orig, ensemble_member=0)
        timeseries.write()

    def test_shrink_timeseries_neq(self):
        # Decrease end date time
        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import_neq", binary=False)
        orig_values = np.array(timeseries.get('S', ensemble_member=0), copy=True)
        orig_times  = timeseries.times
        orig_end    = timeseries.end_datetime
        timeseries.resize(timeseries.start_datetime,
                          timeseries.end_datetime - datetime.timedelta(seconds=3600))
        timeseries.get('S', ensemble_member=0)[-1] = 12345.0
        timeseries.write()

        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import_neq", binary=False)
        self.assertEquals(timeseries.get('S')[-1], 12345.0)
        self.assertEquals(len(timeseries.get('S')), len(orig_values) - 1)
        timeseries.set('S', orig_values, ensemble_member=0)
        # Because we don't support extension of nonequidistant series, we need
        # to reset the times manually.
        timeseries._times = orig_times
        timeseries._end_datetime = orig_end
        timeseries.write()

        # Increase start date time
        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import_neq", binary=False)
        orig_values = np.array(timeseries.get('S', ensemble_member=0), copy=True)
        orig_times  = timeseries.times
        orig_start  = timeseries.start_datetime
        timeseries.resize(timeseries.start_datetime + datetime.timedelta(seconds=7200),
                          timeseries.end_datetime )
        timeseries.get('S', ensemble_member=0)[0] = 0.0
        timeseries.write()

        timeseries = pi.Timeseries(
            self.data_config, data_path(), "timeseries_import_neq", binary=False)
        self.assertEquals(timeseries.get('S')[0], 0.0)
        self.assertEquals(len(timeseries.get('S')), len(orig_values) - 1)
        timeseries.set('S', orig_values, ensemble_member=0)
        # Because we don't support extension of nonequidistant series, we need
        # to reset the times manually.
        timeseries._times = orig_times
        timeseries._start_datetime = orig_start
        timeseries.write()

    def test_placeholder_timeseries(self):
        for binary in [True, False]:
            timeseries = pi.Timeseries(
                self.data_config, data_path(), "timeseries_export", binary=binary)
            self.assertEquals(len(timeseries.get('S', ensemble_member=0)), 10)
