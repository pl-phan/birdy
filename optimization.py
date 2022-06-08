import numpy as np
from scipy.optimize import least_squares

from utils import mul_1d


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

        p_opt = results.x
        p_hes = results.jac.T @ results.jac
        try:
            p_cov = np.linalg.inv(p_hes)
        except np.linalg.LinAlgError:
            p_cov = np.full_like(p_hes, fill_value=float('+inf'))

        p_sig = p_cov.diagonal() ** 0.5
        p_rel = p_sig / p_opt

        return p_opt, p_cov, p_rel


if __name__ == '__main__':
    # TODO TESTS
    pass
