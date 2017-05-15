from unittest import TestCase

import rtctools.data.csv as csv
import os
import numpy as np
from datetime import datetime

from .data_path import data_path


class TestCSV(TestCase):

    def setUp(self):
        pass

    def test_semicolon_separated(self):
        self.lookup_table_0 = csv.load(os.path.join(
            data_path(), 'look_up_table_0.csv'), delimiter=';', with_time=False)
        self.timeseries_import_0 = csv.load(os.path.join(
            data_path(), 'timeseries_import_0.csv'), delimiter=';', with_time=True)
        self.lookup_table_1 = csv.load(os.path.join(
            data_path(), 'look_up_table_1.csv'), delimiter=';', with_time=False)
        self.timeseries_import_1 = csv.load(os.path.join(
            data_path(), 'timeseries_import_1.csv'), delimiter=';', with_time=True)

        self.initial_state_0 = csv.load(os.path.join(
            data_path(), 'initial_state_0.csv'), delimiter=';', with_time=False)
        self.initial_state_1 = csv.load(os.path.join(
            data_path(), 'initial_state_1.csv'), delimiter=';', with_time=False)

    def test_comma_separated(self):
        self.lookup_table_2 = csv.load(os.path.join(
            data_path(), 'look_up_table_2.csv'), delimiter=',', with_time=False)
        self.timeseries_import_3 = csv.load(os.path.join(
            data_path(), 'timeseries_import_2.csv'), delimiter=',', with_time=True)
