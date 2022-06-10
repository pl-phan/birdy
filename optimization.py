from collections import namedtuple

import numpy as np
from scipy.optimize import least_squares

from utils import mul_1d, jac2cov

Results = namedtuple('Results', 'x jac')


class Optimizer:
    def __init__(self, model, measurements, uncertainties):
        self.model = model
        self.measurements = measurements
        self.uncertainties = uncertainties

        self.residuals = None
        self.jacobian = None
        self.current_params = None

    def _propagate(self, params):
        _, x, x_var = self.model(params)

        self.residuals = (x - self.measurements) / self.uncertainties
        self.jacobian = mul_1d(x_var, 1. / self.uncertainties)
        self.current_params = params

    def _get_residuals(self, params):
        self._propagate(params)
        return self.residuals

    def _get_jacobian(self, params):
        if not np.array_equal(params, self.current_params):
            self._propagate(params)
        return self.jacobian

    def least_squares(self, guess, method='variational_eq'):
        if method == 'variational_eq':
            results = least_squares(self._get_residuals, x0=guess, jac=self._get_jacobian,
                                    ftol=1e-6, xtol=5e-5, gtol=float('nan'))
        elif method == 'finite_diff':
            results = least_squares(self._get_residuals, x0=guess, diff_step=1e-3,
                                    ftol=1e-6, xtol=5e-5, gtol=float('nan'))
        else:
            raise NotImplementedError

        return jac2cov(results.x, results.jac)


if __name__ == '__main__':
    # TODO TESTS
    pass
