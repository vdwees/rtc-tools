# cython: embedsignature=True

from rtctools.optimization.optimization_problem import OptimizationProblem

import numpy as np

import logging
logger = logging.getLogger("rtctools")


class TimestepRescalingMixin(OptimizationProblem):
    """
    Hacky temporary mixin for overwriting an equidistant timeseries import with non-equidistant times and values. 

    If ``timestep_rescaling_factor`` is greater than 1, this mixin will become active and rescale the timeseries
    into non-equidistant timeseries. The length of the first timestep will be as close as possible to the equidistant
    timestep, and length of the length of the last timestep will be the length of the first timestep times
    the ``timestep_rescaling_factor``. The timestep length will increase at a linear rate from the first to last timestep.

    The timeseries import values will be overwritten with values interpolated from the original imported timeseries

    :cvar timestep_rescaling_factor: Ratio in timestep size between the first and last timestep. Default is ``1``.
    :cvar debug_timestep_rescaling:  Write scaled values to timeseries import.  Default is ``False``.
    :cvar equidistant_export_times:  Rescale output timeseries to equidistant timesteps.  Default is ``True``.

    Note/TODO: This mixen is not very optimal because it doesn't overwrite the
    behaviour of other mixins, it just executes after them to overwrite data.
    """

    timestep_rescaling_factor = 1
    debug_timestep_rescaling = False
    equidistant_export_times = True

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super(TimestepRescalingMixin, self).__init__(**kwargs)
        
        # A quick check to see if the parent classes have been initialized in the right order
        # (required because this implementation is hacky)
        try:
            mro_list = [class_instance.__name__ for class_instance in type(self).mro()]
            assert(mro_list.index('PIMixin') > mro_list.index('TimestepRescalingMixin'))
        except ValueError:
            raise AssertionError('TimestepRescalingMixin requires PIMixin')
        except AssertionError:
            raise AssertionError('TimestepRescalingMixin must be inherited before PIMixin. Current order is: \n{}'.format(', '.join(mro_list)))


    def pre(self):
        super(TimestepRescalingMixin, self).pre()

        try:
            assert(self.timestep_rescaling_factor >= 1)
        except AssertionError:
            raise NotImplementedError('TimestepRescalingMixin: timestep_rescaling_factor less than 1 has not been tested.')

        # Only apply hack if rescaling factor is greater than 1:
        if self.timestep_rescaling_factor != 1:

            logger.info('TimestepRescalingMixin: Adjusting timeseries timesteps with timestep_rescaling_factor = {}'.format(self.timestep_rescaling_factor))

            # Store timeseries times for use during post() export 
            # (to avoid having to reconfigure FEWS to accept non-equidistant timeseries)
            self.equidistant_datetimes = self._timeseries_import.times[self._timeseries_import.forecast_index:]
            self.equidistant_times = self._datetime_to_sec(self._timeseries_import.times)
            self.equidistant_deltatime = self._timeseries_import._dt

            equidistant_times = self.equidistant_times
            equidistant_timestep = self._timeseries_import._dt.total_seconds()

            # nsteps does not include first timestep (t = 0)
            nsteps = round((equidistant_times[-1] / equidistant_timestep) * 2 / (1 + self.timestep_rescaling_factor))

            spacing_array = np.cumsum(np.linspace(1, self.timestep_rescaling_factor, nsteps)) * equidistant_timestep
            non_equidistant_times = np.insert(spacing_array * equidistant_times[-1] / spacing_array[-1], 0, 0)

            # Overwrite _timeseries_import times with adjusted times
            self._timeseries_import._times = self._sec_to_datetime(non_equidistant_times)

            # Overwrite _timeseries_import values with interpolated values
            for var_name, values in self._timeseries_import.iteritems():
                values = np.interp(non_equidistant_times, equidistant_times, values)
                self._timeseries_import.set(var_name, values)

            # Write changes to time-related variables
            self._timeseries_import._dt = None
            self._timeseries_import_times = self._datetime_to_sec(self._timeseries_import.times)

            logger.info('TimestepRescalingMixin: Done adjusting timeseries timesteps.' +
                        ' First timestep length: {} s.'.format(non_equidistant_times[1]) +
                        ' Last timestep length: {} s.'.format(non_equidistant_times[-1] - non_equidistant_times[-2]))


        # These new timestps can be written to timeseries import if debugging/validation is needed
        if self.debug_timestep_rescaling:
            logger.info('TimestepRescalingMixin: Overwriting Input timeseries.')
            self._timeseries_import.write()

    def post(self):
        super(TimestepRescalingMixin, self).post()

        if self.timestep_rescaling_factor != 1 and self.equidistant_export_times:
            logger.info('TimestepRescalingMixin: Overwriting output timeseries with interpolated equidistant timesteps.')
            self._timeseries_export._times = self.equidistant_datetimes
            self._timeseries_export._dt = self.equidistant_deltatime

            for var_name, values in self._timeseries_export.iteritems():
                values = np.interp(self.equidistant_times, self._timeseries_import_times, values)
                self._timeseries_export.set(var_name, values)
            self._timeseries_export.write()
