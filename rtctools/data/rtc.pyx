# cython: embedsignature=True

import xml.etree.cElementTree as ET
import numpy as np
import logging
import datetime
import struct
import io
import os
import logging
from collections import namedtuple

ts_ids = namedtuple('ids', 'location_id parameter_id qualifier_id')
p_ids = namedtuple('ids', 'model_id location_id parameter_id')

ns = {'fews': 'http://www.wldelft.nl/fews',
      'pi': 'http://www.wldelft.nl/fews/PI'}

logger = logging.getLogger("rtctools")

def pi_parameter_id(parameter_id, location_id=None, model_id=None):
    """
    Convert a model, location and parameter combination to a single parameter id
    of the form model:location:parameter.
    """
    if location_id is not None:
        parameter_id = location_id + ':' + parameter_id
    if model_id is not None:
        parameter_id = model_id + ':' + parameter_id
    return parameter_id

class DataConfig:
    """
    rtcDataConfig wrapper.

    Used to map PI timeseries to RTC-Tools variable names.
    """

    def __init__(self, folder):
        """
        Parse rtcDataConfig file

        :param folder: Folder in which rtcDataConfig.xml is located.
        """
        self._variable_map = {}
        self._location_parameter_ids = {}
        self._parameter_map = {}
        self._model_parameter_ids = {}
        self._basename_import = None
        self._basename_export = None

        path = os.path.join(folder, "rtcDataConfig.xml")
        try:
            tree = ET.parse(path)
            root = tree.getroot()

            timeseriess1 = root.findall('./*/fews:timeSeries', ns)
            timeseriess2 = root.findall('./fews:timeSeries', ns)
            timeseriess1.extend(timeseriess2)

            for timeseries in timeseriess1:
                pi_timeseries = timeseries.find('fews:PITimeSeries', ns)
                if pi_timeseries is not None:
                    self._location_parameter_ids[timeseries.get('id')] = \
                        self._pi_location_parameter_id(pi_timeseries, 'fews')
                    self._variable_map[self._pi_timeseries_id(
                        pi_timeseries, 'fews')] = timeseries.get('id')

            for k in ['import', 'export']:
                res = root.find(
                    './fews:%s/fews:PITimeSeriesFile/fews:timeSeriesFile' % k, ns)
                if res is not None:
                    setattr(self, 'basename_%s' %
                            k, os.path.splitext(res.text)[0])

            parameters = root.findall('./fews:parameter', ns)
            if parameters is not None:
                for parameter in parameters:
                    pi_parameter = parameter.find('fews:PIParameter', ns)
                    if pi_parameter is not None:
                        self._model_parameter_ids[parameter.get('id')] = \
                            self._pi_model_parameter_id(pi_parameter, 'fews')
                        self._parameter_map[self._pi_parameter_id(
                             pi_parameter, 'fews')] = parameter.get('id')

        except IOError:
            logger.error(
                'No rtcDataConfig.xml file was found in "{}".'.format(folder))
            raise

    def _pi_timeseries_id(self, el, namespace):
        location_id = el.find(namespace + ':locationId', ns).text
        parameter_id = el.find(namespace + ':parameterId', ns).text

        timeseries_id = location_id + ':' + parameter_id

        qualifiers = el.findall(namespace + ':qualifierId', ns)
        qualifier_ids = []
        for qualifier in qualifiers:
            qualifier_ids.append(qualifier.text)

        if len(qualifier_ids) > 0:
            qualifier_ids.sort()

            return timeseries_id + ':' + ':'.join(qualifier_ids)
        else:
            return timeseries_id

    def _pi_location_parameter_id(self, el, namespace):
        qualifier_ids = []
        qualifiers = el.findall(namespace + ':qualifierId', ns)
        for qualifier in qualifiers:
            qualifier_ids.append(qualifier.text)

        location_parameter_ids = ts_ids(location_id  = el.find(namespace + ':locationId', ns).text,
                                        parameter_id = el.find(namespace + ':parameterId', ns).text,
                                        qualifier_id = qualifier_ids)
        return location_parameter_ids

    def _pi_parameter_id(self, el, namespace):
        model_id = el.find(namespace + ':modelId', ns).text
        location_id = el.find(namespace + ':locationId', ns).text
        parameter_id = el.find(namespace + ':parameterId', ns).text

        return pi_parameter_id(parameter_id, location_id, model_id)

    def _pi_model_parameter_id(self, el, namespace):
        model_id = el.find(namespace + ':modelId', ns).text
        location_id = el.find(namespace + ':locationId', ns).text
        parameter_id = el.find(namespace + ':parameterId', ns).text

        model_parameter_ids = p_ids(model_id  = (model_id if model_id is not None else ""),
                                    location_id = (location_id if location_id is not None else ""),
                                    parameter_id = (parameter_id if parameter_id is not None else ""))

        return model_parameter_ids

    def variable(self, pi_header):
        """
        Map a PI timeseries header to a RTC-Tools timeseries ID.

        :param pi_header: XML ElementTree node containing a PI timeseries header.

        :returns: A timeseries ID.
        :rtype: string
        """
        series_id = self._pi_timeseries_id(pi_header, 'pi')
        try:
            return self._variable_map[series_id]
        except KeyError:
            return series_id

    def location_parameter_id(self, variable):
        """
        Map a RTC-Tools timeseries ID to a named tuple of location, parameter
        and qualifier ID's.

        :param variable: A timeseries ID.

        :returns: A named tuple with fields location_id, parameter_id and qualifier_id.
        :rtype: namedtuple
        :raises KeyError: If the timeseries ID has no mapping in rtcDataConfig.
        """
        return self._location_parameter_ids[variable]

    def parameter(self, parameter_id, location_id=None, model_id=None):
        """
        Map a combination of parameter ID, location ID, model ID to a
        RTC-Tools parameter ID.

        :param parameter_id: String with parameter ID
        :param location_id: String with location ID
        :param model_id: String with model ID

        :returns: A parameter ID.
        :rtype: string
        :raises KeyError: If the combination has no mapping in rtcDataConfig.
        """
        parameter_id_long = pi_parameter_id(parameter_id, location_id, model_id)

        return self._parameter_map[parameter_id_long]

    def model_parameter_id(self, parameter):
        """
        Map a RTC-Tools model parameter ID to a named tuple of model, location
        and parameter ID's.

        :param parameter: A model parameter ID.

        :returns: A named tuple with fields model_id, location_id and parameter_id.
        :rtype: namedtuple
        :raises KeyError: If the paramter ID has no mapping in rtcDataConfig.
        """
        return self._model_parameter_ids[parameter]
