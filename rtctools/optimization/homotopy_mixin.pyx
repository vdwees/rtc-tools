# cython: embedsignature=True

from rtctools.optimization.optimization_problem import OptimizationProblem
from rtctools.optimization.timeseries import Timeseries
import logging

logger = logging.getLogger("rtctools")


class HomotopyMixin(OptimizationProblem):
    """
    Adds homotopy to your optimization problem.  A homotopy is a continuous transformation between
    two optimization problems, parametrized by a single parameter :math:`\\theta \\in [0, 1]`.

    Homotopy may be used to solve non-convex optimization problems, by starting with a convex 
    approximation at :math:`\\theta = 0.0` and ending with the non-convex problem at 
    :math:`\\theta = 1.0`.

    .. note::

        It is advised to look for convex reformulations of your problem, before resorting to a use
        of the (potentially expensive) homotopy process.

    """

    def seed(self, ensemble_member):
        seed = super(HomotopyMixin, self).seed(ensemble_member)
        if self._theta > 0:
            # Add previous results to seed
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
        """
        Returns a dictionary of options controlling the homotopy process.

        +------------------------+------------+---------------+
        | Option                 | Type       | Default value |
        +========================+============+===============+
        | ``delta_theta_0``      | ``float``  | ``1.0``       |
        +------------------------+------------+---------------+
        | ``delta_theta_min``    | ``float``  | ``0.01``      |
        +------------------------+------------+---------------+
        | ``homotopy_parameter`` | ``string`` | ``theta``     |
        +------------------------+------------+---------------+

        The homotopy process is controlled by the homotopy parameter in the model, specified
        by the option ``homotopy_parameter``.  The homotopy parameter is initialized to ``0.0``,
        and increases to a value of ``1.0`` with a dynamically changing step size.  This step
        size is initialized with the value of the option ``delta_theta_0``.  If this step
        size is too large, i.e., if the problem with the increased homotopy parameter fails to
        converge, the step size is halved.  The process of halving terminates when the step size falls
        below the minimum value specified by the option ``delta_theta_min``.

        :returns: A dictionary of homotopy options.
        """

        return {'delta_theta_0'     : 1.0,
                'delta_theta_min'   : 0.01,
                'homotopy_parameter': 'theta'}

    def optimize(self, preprocessing=True, postprocessing=True):
        # Pre-processing
        self.pre()

        # Homotopy loop
        self._theta = 0.0

        options = self.homotopy_options()
        delta_theta = options['delta_theta_0']

        while self._theta <= 1.0:
            logger.info("Solving with homotopy parameter theta = {}.".format(self._theta))

            success = super(HomotopyMixin, self).optimize(preprocessing=False, postprocessing=False)
            if success:
                self._results = [self.extract_results(ensemble_member) for ensemble_member in range(self.ensemble_size)]
            else:
                if self._theta == 0.0:
                    break

                self._theta -= delta_theta
                delta_theta /= 2

                if delta_theta < options['delta_theta_min']:
                    break

            self._theta += delta_theta

        # Post-processing
        self.post()

        return success
