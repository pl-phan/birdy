import numpy as np
import plotly.graph_objects as go
from scipy.optimize import least_squares


class Optimizer:
    def __init__(self, model, measurements, uncertainties):
        self.model = model
        self.measurements = measurements
        self.uncertainties = uncertainties

        self.residuals = np.empty_like(measurements)
        self.jacobian = np.empty_like(measurements)
        self.current_mu = None

    def _propagate(self, mu):
        x, x_var = self.model(mu)

        self.residuals = (x - self.measurements) / self.uncertainties
        self.jacobian = x_var / self.uncertainties
        self.current_mu = mu

    def _get_residuals(self, params):
        mu, = params
        self._propagate(mu)
        return self.residuals

    def _get_jacobian(self, params):
        mu, = params
        if mu != self.current_mu:
            self._propagate(mu)
        return np.expand_dims(self.jacobian, axis=1)

    def least_squares(self, mu0, method='variational_eq'):
        if method == 'variational_eq':
            return least_squares(self._get_residuals, x0=mu0, jac=self._get_jacobian,
                                 ftol=1e-6, xtol=5e-5, gtol=float('nan'))
        elif method == 'finite_diff':
            return least_squares(self._get_residuals, x0=mu0, diff_step=1e-5,
                                 ftol=1e-6, xtol=5e-5, gtol=float('nan'))
        else:
            raise NotImplementedError


if __name__ == '__main__':
    # TODO TESTS
    pass
