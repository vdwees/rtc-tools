# cython: embedsignature=True

from rtctools.optimization.optimization_problem import OptimizationProblem
from rtctools.optimization.timeseries import Timeseries
import logging

logger = logging.getLogger("rtctools")


class HomotopyMixin(OptimizationProblem):
    """
    Homotopy.
    """

    def seed(self, ensemble_member):
        if self._theta == 0.0:
            seed = super(HomotopyMixin, self).seed(ensemble_member)
        else:
            # Seed with previous results
            seed = {}
            for key in self._results[ensemble_member].keys():
                seed[key] = Timeseries(self.times(key), self._results[
                                       ensemble_member][key])
        return seed

    def parameters(self, ensemble_member):
        parameters = super(HomotopyMixin, self).parameters(ensemble_member)

        options = self.homotopy_options()
        parameters[options['homotopy_parameter']] = self._theta

        return parameters

    def homotopy_options(self):
        return {'delta_theta_0'     : 1.0,
                'delta_theta_min'   : 0.01,
                'homotopy_parameter': 'theta'}

    def optimize(self):
        self._theta = 0.0 # Homotopy parameter

        options = self.homotopy_options()
        delta_theta = options['delta_theta_0']

        while self._theta <= 1.0:
            logger.info("Solving with homotopy parameter theta = {}.".format(self._theta))

            success = super(HomotopyMixin, self).optimize()
            if success:
                self._results = [self.extract_results(ensemble_member) for ensemble_member in range(self.ensemble_size)]
            else:
                if self._theta == 0.0:
                    return False

                self._theta -= delta_theta
                delta_theta /= 2

                if delta_theta < options['delta_theta_min']:
                    return False

            self._theta += delta_theta

        return True
