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

ids = namedtuple('ids', 'location_id parameter_id qualifier_id')

ns = {'fews': 'http://www.wldelft.nl/fews',
      'pi': 'http://www.wldelft.nl/fews/PI'}

logger = logging.getLogger("rtctools")


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
        self._map = {}
        self._location_parameter_ids = {}
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
                if pi_timeseries != None:
                    self._location_parameter_ids[timeseries.get('id')] = \
                        self._pi_location_parameter_id(pi_timeseries, 'fews')
                    self._map[self._pi_timeseries_id(
                        pi_timeseries, 'fews')] = timeseries.get('id')

            for k in ['import', 'export']:
                res = root.find(
                    './fews:%s/fews:PITimeSeriesFile/fews:timeSeriesFile' % k, ns)
                if res is not None:
                    setattr(self, 'basename_%s' %
                            k, os.path.splitext(res.text)[0])

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

        location_parameter_ids = ids( location_id  = el.find(namespace + ':locationId', ns).text,
                                      parameter_id = el.find(namespace + ':parameterId', ns).text,
                                      qualifier_id = qualifier_ids)
        return location_parameter_ids

    def variable(self, pi_header):
        """
        Map a PI timeseries header to an RTC-Tools timeseries ID.

        :param pi_header: XML ElementTree node containing a PI timeseries header.

        :returns: A timeseries ID.
        :rtype: string
        """
        series_id = self._pi_timeseries_id(pi_header, 'pi')
        try:
            return self._map[series_id]
        except KeyError:
            return series_id

    def location_parameter_id(self, variable):
        """
        Map a RTC-Tools timeseries ID to a named tuple of location, parameter
        and qualifier ID's.

        :param variable: A timeseries ID.

        :returns: A named tuple with fields location_id, parameter_id and qualifier_id.
        :rtype: namedtuple
        :raises KeyError: If the timeseries Id has no mapping in rtcDataConfig.
        """
        return self._location_parameter_ids[variable]
