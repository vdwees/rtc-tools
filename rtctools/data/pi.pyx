# cython: embedsignature=True

import xml.etree.cElementTree as ET
import numpy as np
import datetime
import struct
import io
import os
import logging
import copy
import bisect

ns = {'fews': 'http://www.wldelft.nl/fews',
      'pi': 'http://www.wldelft.nl/fews/PI'}


def _parse_date_time(el):
    # Parse a PI date time element.
    return datetime.datetime.strptime(el.get('date') + ' ' + el.get('time'), '%Y-%m-%d %H:%M:%S')


def _parse_time_step(el):
    # Parse a PI time step element.
    if el.get('unit') == 'second':
        return datetime.timedelta(seconds=int(el.get('multiplier')))
    elif el.get('unit') == 'nonequidistant':
        return None
    else:
        raise Exception('Unsupported unit type: ' + el.get('unit'))


def _floor_date_time(dt=datetime.datetime.now(), tdel=datetime.timedelta(minutes=1)):
    # Floor a PI date time based on a PI time step
    roundTo = tdel.total_seconds()

    seconds = (dt - dt.min).seconds
    # // is a floor division:
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


class Diag:
    """
    diag wrapper.
    """

    ERROR_FATAL = 1 << 0
    ERROR = 1 << 1
    WARN = 1 << 2
    INFO = 1 << 3
    DEBUG = 1 << 4

    def __init__(self, folder, basename='diag'):
        """
        Parse diag file.

        :param folder:   Folder in which diag.xml is found or to be created.
        :param basename: Alternative basename for the diagnostics XML file.
        """
        self._path_xml = os.path.join(folder, basename + '.xml')

        self._tree = ET.parse(self._path_xml)
        self._xml_root = self._tree.getroot()

    def get(self, level=ERROR_FATAL):
        """
        Return only the wanted levels (debug, info, etc.)

        :param level: Log level.
        """

        diag_lines = self._xml_root.findall("*", ns)
        diag_lines_out = []
        USED_LEVELS = []

        if level & self.ERROR_FATAL:
            USED_LEVELS.append('0')
        if level & self.ERROR:
            USED_LEVELS.append('1')
        if level & self.WARN:
            USED_LEVELS.append('2')
        if level & self.INFO:
            USED_LEVELS.append('3')
        if level & self.DEBUG:
            USED_LEVELS.append('4')

        for child in diag_lines:
            for used_level in USED_LEVELS:
                if child.get('level') == used_level:
                    diag_lines_out.append(child)

        return diag_lines_out

    @property
    def has_errors(self):
        """
        True if the log contains errors.
        """
        error_levels = self.ERROR_FATAL | self.ERROR
        diag_lines = self.get(error_levels)

        if len(diag_lines) > 0:
            return True
        else:
            return False


class DiagHandler(logging.Handler):
    """
    PI diag file logging handler.
    """

    def __init__(self, folder, basename='diag', level=logging.NOTSET):
        super(DiagHandler,  self).__init__(level=level)

        self._path_xml = os.path.join(folder, basename + '.xml')

        try:
            self._tree = ET.parse(self._path_xml)
            self._xml_root = self._tree.getroot()
        except:
            self._xml_root = ET.Element('{%s}Diag' % (ns['pi'], ))
            self._tree = ET.ElementTree(element=self._xml_root)

        self._map_level = {50: 0, 40: 1, 30: 2, 20: 3, 10: 4, 0: 4}

    def emit(self, record):
        self.format(record)

        self.acquire()
        el = ET.SubElement(self._xml_root, '{%s}line' % (ns['pi'], ))
        # Work around cElementTree issue 21403
        el.set('description', record.message)
        el.set('eventCode', record.module + '.' + record.funcName)
        el.set('level', str(self._map_level[record.levelno]))
        self.release()

    def append_element(self, el):
        self.acquire()
        self._xml_root.append(el)
        self.release()

    def flush(self):
        self._tree.write(self._path_xml)

    def close(self):
        self.flush()
        super(DiagHandler, self).close()


class ParameterConfig:
    """
    rtcParameterConfig wrapper.
    """

    def __init__(self, folder, basename):
        """
        Parses a rtcParameterConfig file.

        :param folder:    Folder in which the parameter configuration file is located.
        :param basename:  Basename of the parameter configuration file (e.g, 'rtcParameterConfig').
        """
        self._path_xml = os.path.join(folder, basename + '.xml')

        self._tree = ET.parse(self._path_xml)
        self._xml_root = self._tree.getroot()

    def get(self, group_id, parameter_id, location_id=None, model=None):
        """
        Returns the value of the parameter with ID parameter_id in the group with ID group_id.

        :param group_id:     The ID of the parameter group to look in.
        :param parameter_id: The ID of the parameter to look for.
        :param location_id:  The optional  ID of the parameter location to look in.
        :param model:        The optional ID of the parameter model to look in.

        :returns: The value of the specified parameter.
        """
        groups = self._xml_root.findall(
            "pi:group[@id='{}']".format(group_id), ns)
        for group in groups:
            el = group.find("pi:locationId", ns)
            if location_id != None and el != None:
                if location_id != el.text:
                    continue

            el = group.find("pi:model", ns)
            if model != None and el != None:
                if model != el.text:
                    continue

            el = group.find("pi:parameter[@id='{}']".format(parameter_id), ns)
            if el == None:
                raise KeyError
            return self._parse_parameter(el)

        raise KeyError("No such parameter ({}, {})".format(
            group_id, parameter_id))

    def set(self, group_id, parameter_id, new_value, location_id=None, model=None):
        """
        Set the value of the parameter with ID parameter_id in the group with ID group_id.

        :param group_id:     The ID of the parameter group to look in.
        :param parameter_id: The ID of the parameter to look for.
        :param new_value:    The new value for the parameter.
        :param location_id:  The optional  ID of the parameter location to look in.
        :param model:        The optional ID of the parameter model to look in.
        """
        groups = self._xml_root.findall(
            "pi:group[@id='{}']".format(group_id), ns)
        for group in groups:
            el = group.find("pi:locationId", ns)
            if location_id != None and el != None:
                if location_id != el.text:
                    continue

            el = group.find("pi:model", ns)
            if model != None and el != None:
                if model != el.text:
                    continue

            el = group.find("pi:parameter[@id='{}']".format(parameter_id), ns)
            if el == None:
                raise KeyError
            for child in el:
                if child.tag.endswith('boolValue'):
                    if new_value == True:
                        child.text = 'true'
                        return
                    elif new_value == False:
                        child.text = 'false'
                        return
                    else:
                        raise Exception(
                            "Unsupported value for tag {}".format(child.tag))
                elif child.tag.endswith('intValue'):
                    child.text = str(int(new_value))
                    return
                elif child.tag.endswith('dblValue'):
                    child.text = str(new_value)
                    return
                else:
                    raise Exception("Unsupported tag {}".format(child.tag))

        raise KeyError("No such parameter ({}, {})".format(
            group_id, parameter_id))

    def write(self):
        """
        Writes the parameter configuration to a file.
        """
        self._tree.write(self._path_xml)

    def _parse_type(self, fews_type):
        # Parse a FEWS type to an np type
        if fews_type == 'double':
            return np.dtype('float')
        else:
            return np.dtype('S128')

    def _parse_parameter(self, parameter):
        child = parameter[0]
        if child.tag.endswith('boolValue'):
            if child.text.lower() == 'true':
                return True
            else:
                return False
        elif child.tag.endswith('intValue'):
            return int(child.text)
        elif child.tag.endswith('dblValue'):
            return float(child.text)
        elif child.tag.endswith('stringValue'):
            return child.text
        # return dict of lisstart_datetime
        elif child.tag.endswith('table'):
            columnId = {}
            columnType = {}
            for key in child.find("pi:row", ns).attrib:
                # default Id
                columnId[key] = key
                columnType[key] = np.dtype(
                    'S128')                     # default Type

            # get Id's if present
            el_columnIds = child.find("pi:columnIds", ns)
            if el_columnIds != None:
                for key, value in el_columnIds.attrib.iteritems():
                    columnId[key] = value

            # get Types if present
            el_columnTypes = child.find("pi:columnTypes", ns)
            if el_columnTypes != None:
                for key, value in el_columnTypes.attrib.iteritems():
                    columnType[key] = self._parse_type(value)

            # get table contenstart_datetime
            el_row = child.findall("pi:row", ns)
            table = {columnId[key]: np.empty(len(el_row),            # initialize table
                                             columnType[key]) for key in columnId}

            i_row = 0
            for row in el_row:
                for key, value in row.attrib.iteritems():
                    table[columnId[key]][i_row] = value
                i_row += 1
            return table
        else:
            raise Exception("Unsupported tag {}".format(child.tag))

    def __iter__(self):
        # Iterate over all parameter key, value pairs.
        groups = self._xml_root.findall("pi:group", ns)
        for group in groups:
            parameters = group.findall("pi:parameter", ns)
            for parameter in parameters:
                yield parameter.attrib['id'], self._parse_parameter(parameter)


class Timeseries:
    """
    PI timeseries wrapper.
    """

    def __init__(self, data_config, folder, basename, binary=True):
        """
        Load the timeseries from disk.

        :param data_config:      A :class:`DataConfig` object.
        :param folder:           The folder in which the time series is located.
        :param basename:         The basename of the time series file.
        :param binary:           True if the time series data is stored in a separate binary file.
        """
        self._data_config = data_config

        self._folder = folder
        self._basename = basename

        self._path_xml = os.path.join(self._folder, basename + '.xml')

        self._internal_dtype = np.float64
        self._pi_dtype = np.float32

        self._tree = ET.parse(self._path_xml)
        self._xml_root = self._tree.getroot()

        self._values = [{}]

        self._binary = binary

        f = None
        if self._binary:
            try:
                f = io.open(self.binary_path, 'rb')
            except IOError:
                # Support placeholder XML files.
                pass

        # Check data consistency
        self._dt = None
        self._start_datetime = None
        self._end_datetime = None
        self._forecast_datetime = None
        self._forecast_index = None
        for series in self._xml_root.findall('pi:series', ns):
            header = series.find('pi:header', ns)

            variable = self._data_config.variable(header)

            dt = _parse_time_step(header.find('pi:timeStep', ns))
            if self._dt == None:
                self._dt = dt
            else:
                if dt != self._dt:
                    raise Exception(
                        'PI: Not all timeseries share the same time step size')
            try:
                start_datetime = _parse_date_time(
                    header.find('pi:startDate', ns))
                if self._start_datetime == None:
                    self._start_datetime = start_datetime
                else:
                    if start_datetime < self._start_datetime:
                        self._start_datetime = start_datetime
            except (AttributeError, ValueError):
                raise Exception('PI: Variable {} in {} has no startDate.'.format(
                    variable, os.path.join(self._folder, basename + '.xml')))

            try:
                end_datetime = _parse_date_time(header.find('pi:endDate', ns))
                if self._end_datetime == None:
                    self._end_datetime = end_datetime
                else:
                    if end_datetime > self._end_datetime:
                        self._end_datetime = end_datetime
            except (AttributeError, ValueError):
                raise Exception('PI: Variable {} in {} has no endDate.'.format(
                    variable, os.path.join(self._folder, basename + '.xml')))

            el = header.find('pi:forecastDate', ns)
            if el != None:
                forecast_datetime = _parse_date_time(el)
            # the timeseries has no forecastDate, so the forecastDaend_datetime
            # is set to the startDaend_datetime (per the PI-schema)
            else:
                forecast_datetime = start_datetime
            if self._forecast_datetime == None:
                self._forecast_datetime = forecast_datetime
            else:
                if forecast_datetime != self._forecast_datetime:
                    raise Exception(
                        'PI: Not all timeseries share the same forecastDate')

        # Define the times, and floor the global forecast_datetime to the
        # global time step to get its index
        if self._dt:
            self._len = int(round(
                (self._end_datetime - self._start_datetime).total_seconds() / self._dt.total_seconds() + 1))
            self._times = [self._start_datetime + i *
                           self._dt for i in range(0, self._len)]
        else: # Timeseries are non-equidistant
            self._len = 0
            for series in self._xml_root.findall('pi:series', ns):
                events = series.findall('pi:event', ns)
                # We assume that timeseries can differ in length, but always are a complete
                # 'slice' of datetimes between start and end. The longest timeseries then
                # containts all datetimes between start and end.
                if len(events) > self._len:
                    self._len = len(events)
                    self._times = []
                    for i in range(len(events)):
                        self._times.append(_parse_date_time(events[i]))

        if self._forecast_datetime != None:
            if self._dt:
                self._forecast_datetime = _floor_date_time(
                    dt=self._forecast_datetime, tdel=self._dt)
            try:
                self._forecast_index = self._times.index(
                    self._forecast_datetime)
            except ValueError:
                # This may occur if forecast_datetime is outside of
                # the timeseries' range.  Can be a valid case for historical
                # timeseries, for instance.
                self._forecast_index = -1

        # Parse data
        for series in self._xml_root.findall('pi:series', ns):
            header = series.find('pi:header', ns)

            variable = self._data_config.variable(header)

            dt = _parse_time_step(header.find('pi:timeStep', ns))
            start_datetime = _parse_date_time(header.find('pi:startDate', ns))
            end_datetime = _parse_date_time(header.find('pi:endDate', ns))

            el = header.find('pi:ensembleMemberIndex', ns)
            if el != None:
                ensemble_member = int(el.text)
                while ensemble_member >= len(self._values):
                    self._values.append({})
            else:
                ensemble_member = 0

            if self._dt:
                n_values = int(
                    round((end_datetime - start_datetime).total_seconds() / dt.total_seconds() + 1))
            else:
                n_values = len(self._times)

            if self._binary:
                if f != None:
                    self._values[ensemble_member][variable] = np.fromstring(
                        f.read(self._pi_dtype(0).itemsize * n_values), dtype=self._pi_dtype)
                else:
                    self._values[ensemble_member][variable] = np.empty(
                        n_values, dtype=self._internal_dtype)
                    self._values[ensemble_member][variable].fill(np.nan)
            else:
                events = series.findall('pi:event', ns)
                self._values[ensemble_member][variable] = np.empty(
                    n_values, dtype=self._internal_dtype)
                self._values[ensemble_member][variable].fill(np.nan)
                for i in range(min(n_values, len(events))):
                    self._values[ensemble_member][variable][
                        i] = float(events[i].get('value'))

            miss_val = float(header.find('pi:missVal', ns).text)
            self._values[ensemble_member][variable][self._values[
                ensemble_member][variable] == miss_val] = np.nan

            # Prepend empty space, if start_datetime > self._start_datetime.
            if start_datetime > self._start_datetime:
                filler = np.empty(int(round(
                    (start_datetime - self._start_datetime).total_seconds() / dt.total_seconds())), dtype=self._internal_dtype)
                filler.fill(np.nan)
                self._values[ensemble_member][variable] = np.hstack(
                    (filler, self._values[ensemble_member][variable]))

            # Append empty space, if end_datetime < self._end_datetime
            if end_datetime < self._end_datetime:
                if self._dt:
                    filler = np.empty(int(round(
                        (self._end_datetime - end_datetime).total_seconds() / dt.total_seconds())), dtype=self._internal_dtype)
                else:
                    filler = np.empty(int(round(
                        bisect.bisect_left(self._times, self._end_datetime) - bisect.bisect_left(self._times, end_datetime))), dtype=self._internal_dtype)

                filler.fill(np.nan)
                self._values[ensemble_member][variable] = np.hstack(
                    (self._values[ensemble_member][variable], filler))

        if f != None and self._binary:
            f.close()

    def write(self):
        """
        Writes the time series data to disk.
        """
        if self._binary:
            f = io.open(self.binary_path, 'wb')

        for ensemble_member in range(len(self._values)):
            for series in self._xml_root.findall('pi:series', ns):
                header = series.find('pi:header', ns)

                # Updaend_datetime the time range, which may have changed
                el = header.find('pi:startDate', ns)
                el.set('date', self._start_datetime.strftime('%Y-%m-%d'))
                el.set('time', self._start_datetime.strftime('%H:%M:%S'))

                el = header.find('pi:endDate', ns)
                el.set('date', self._end_datetime.strftime('%Y-%m-%d'))
                el.set('time', self._end_datetime.strftime('%H:%M:%S'))

                el = header.find('pi:ensembleMemberIndex', ns)
                if el != None:
                    el.text = str(ensemble_member)

                variable = self._data_config.variable(header)

                miss_val = float(header.find('pi:missVal', ns).text)
                l = self._values[ensemble_member][variable]

                # Replace NaN with missing value
                nans = np.isnan(l)
                l[nans] = miss_val

                # Write output
                if self._binary:
                    f.write(l.astype(self._pi_dtype).tostring())
                else:
                    events = series.findall('pi:event', ns)

                    t = self._start_datetime
                    for i in range(min(len(events), len(l))):
                        # Set the date/time, so that any date/time steps that
                        # are wrong in the placeholder file are corrected.
                        events[i].set('date', t.strftime('%Y-%m-%d'))
                        events[i].set('time', t.strftime('%H:%M:%S'))

                        # Set the value
                        events[i].set('value', str(l[i]))
                        if self.dt:
                            t += self.dt
                        else:
                            t = self.times[i]
                    for i in range(len(events), len(l)):
                        event = ET.Element('pi:event')
                        event.set('date', t.strftime('%Y-%m-%d'))
                        event.set('time', t.strftime('%H:%M:%S'))
                        event.set('value', str(l[i]))
                        series.append(event)
                        if self.dt:
                            t += self.dt
                        else:
                            t = self.times[i]

                # Restore NaN
                l[nans] = np.nan

        if self._binary:
            f.close()

        self._tree.write(self._path_xml)

    @property
    def start_datetime(self):
        """
        Start time.
        """
        return self._start_datetime

    @property
    def end_datetime(self):
        """
        End time.
        """
        return self._end_datetime

    @property
    def forecast_datetime(self):
        """
        Forecast time (t0).
        """
        return self._forecast_datetime

    @property
    def forecast_index(self):
        """
        Forecast time (t0) index.
        """
        return self._forecast_index

    @property
    def dt(self):
        """
        Time step.
        """
        return self._dt

    @property
    def times(self):
        """
        Time stamps.
        """
        return self._times

    def get(self, variable, ensemble_member=0):
        """
        Look up a time series.

        :param variable:        Time series ID.
        :param ensemble_member: Ensemble member index.

        :returns: A :class:`Timeseries` object.
        """
        return self._values[ensemble_member][variable]

    def set(self, variable, new_values, ensemble_member=0):
        """
        Fill a time series with new values.

        :param variable:        Times eries ID.
        :param new_values:      List of new values.
        :param ensemble_member: Ensemble member index.
        """
        self._values[ensemble_member][variable] = new_values

    def resize(self, start_datetime, end_datetime):
        """
        Resize the timeseries to stretch from start_datetime to end_datetime.

        :param start_datetime: Start date and time.
        :param end_datetime:   End date and time.
        """

        if self._dt:
            n_delta_s = int(round(
                (start_datetime - self._start_datetime).total_seconds() / self._dt.total_seconds()))
        else:
            n_delta_s = bisect.bisect_left(self._times, start_datetime) - \
                bisect.bisect_left(self._times, self._start_datetime)

        for ensemble_member in range(len(self._values)):
            if n_delta_s > 0:
                for key in self._values[ensemble_member].keys():
                    self._values[ensemble_member][key] = self._values[
                        ensemble_member][key][n_delta_s:]
            elif n_delta_s < 0:
                filler = np.empty(abs(n_delta_s))
                filler.fill(np.nan)
                for key in self._values[ensemble_member].keys():
                    self._values[ensemble_member][key] = np.hstack(
                        (filler, self._values[ensemble_member][key]))
        self._start_datetime = start_datetime

        if self._dt:
            n_delta_e = int(round(
                (end_datetime - self._end_datetime).total_seconds() / self._dt.total_seconds()))
        else:
            n_delta_e = bisect.bisect_left(self._times, end_datetime) - \
                bisect.bisect_left(self._times, self._end_datetime)
        for ensemble_member in range(len(self._values)):
            if n_delta_e > 0:
                filler = np.empty(n_delta_e)
                filler.fill(np.nan)
                for key in self._values[ensemble_member].keys():
                    self._values[ensemble_member][key] = np.hstack(
                        (self._values[ensemble_member][key], filler))
            elif n_delta_e < 0:
                for key in self._values[ensemble_member].keys():
                    self._values[ensemble_member][key] = self._values[
                        ensemble_member][key][:n_delta_e]
        self._end_datetime = end_datetime

    @property
    def binary_path(self):
        """
        The path for the binary data .bin file.
        """
        return os.path.join(self._folder, self._basename + '.bin')
