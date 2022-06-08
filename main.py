import numpy as np
import plotly.graph_objects as go
from numpy import pi
from plotly.subplots import make_subplots

from integration import integrate
from measurement import measure, add_noise
from optimization import Optimizer
from utils import unit_vector, show_covariance

# physics parameters
G = 6.6743e-11  # m3/kg/s2

# observer parameters
alpha = 170. * pi / 180.  # rad
beta = -3. * pi / 180.  # rad
delta = 4.5e11  # m
earth = np.array((*unit_vector(alpha, beta), 0., 0., 0.)) * delta

# measurement  noise
ranging_noise = 1.1e-7  # s
doppler_noise = 2.4e-2  # Hz


def get_relative_uncertainty(gravity_field, vel, b_sat, seed=None, verbose=False):
    # TODO: plot & print
    # TODO: choose mode (cubesat), data (ranging/doppler), Jn, only variation
    # todo test variational <-> diffs for Jn

    def model(params):
        t, traj, traj_var = integrate(params, vel, b_sat)
        y, y_var = measure(traj - earth, traj_var)
        return t, y, y_var

    if verbose:
        msg = 'true gravitational field: GM={:.7E}'.format(gravity_field[0])
        for i, j in enumerate(gravity_field[1:], 2):
            msg += ' J{:d}={:.7E}'.format(i, j)
        print(msg)

    # data generation with noise
    t_data, y_data, _ = model(gravity_field)
    y_data, uncertainties = add_noise(y_data, ranging_noise, doppler_noise, seed=seed)
    n = len(t_data)

    t_model, y_model, _ = model(gravity_field)
    r_model = y_model - y_data
    if not np.array_equal(t_model, t_data):
        raise ValueError('time indexes do not match')

    p0 = gravity_field * 2
    _, y0, _ = model(p0)
    r0 = y0 - y_data

    # least squares fit
    opt = Optimizer(model, y_data, uncertainties)
    p_opt, p_cov, p_rel = opt.least_squares(p0, method='variational_eq')

    _, y_opt, _ = model(p_opt)
    r_opt = y_model - y_data

    if verbose:
        msg = 'found gravitational field:\nGM={:.7E} ± {:.3f}%'.format(p_opt[0], p_rel[0])
        for i, (j, s) in enumerate(zip(p_opt[1:], p_rel[1:]), 2):
            msg += '\nJ{:d}={:.7E} ± {:.3f}%'.format(i, j, 100. * s)
        print(msg)

        gravity_coefs = ['mu'] + ['J{:d}'.format(i) for i, _ in enumerate(p_rel[1:], 2)]
        gravity_units = ['m3/s2'] + ['' for _ in p_rel[1:]]
        show_covariance(p_opt, p_cov, gravity_coefs, gravity_units, true_values=gravity_field)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02, subplot_titles=('ranging', 'doppler'))
        for r, name in zip((r_model, r0, r_opt), ('model', 'guess', 'solution')):
            fig.add_trace(go.Scatter(x=t_data, y=r[:n], mode='markers', name=name), row=1, col=1)
            fig.add_trace(go.Scatter(x=t_data, y=r[n:], mode='markers', name=name), row=2, col=1)
        fig.show()

    return p_rel


if __name__ == '__main__':
    sig = get_relative_uncertainty(gravity_field=np.array((G * 1.7e18, 1.9e-2)),
                                   vel=15e3, b_sat=3000e3, seed=19960319, verbose=True)
