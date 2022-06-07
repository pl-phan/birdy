import numpy as np
import plotly.graph_objects as go
from numpy import pi

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
ranging_noise = 2e-8  # s
doppler_noise = 5e-3  # Hz


def run(mass_true, vel, b_sat, seed=None, verbose=False):
    # TODO: choose mode (cubesat), data (ranging/doppler), Jn
    # TODO: plot

    def model(mu):
        t, traj, traj_var = integrate(mu, vel, b_sat)
        y, y_var = measure(traj - earth, traj_var)
        return y, y_var

    mu_true = mass_true * G

    # data generation with noise
    y_data, _ = model(mu_true)
    y_data, sigmas = add_noise(y_data, ranging_noise, doppler_noise, seed=seed)

    # least squares fit
    opt = Optimizer(model, y_data, sigmas)
    results = opt.least_squares(mu_true, method='variational_eq')
    # results = opt.least_squares(mu_true, method='finite_diff')
    mu_opt = results.x[0]
    try:
        cov = np.linalg.inv(results.jac.T @ results.jac)
    except np.linalg.LinAlgError:
        cov = np.full((1, 1), fill_value=float('+inf'))
    mu_sig = (cov.diagonal() ** 0.5)[0]

    # p_opt = results.x
    # try:
    #     p_cov = np.linalg.inv(results.jac.T @ results.jac)
    # except np.linalg.LinAlgError:
    #     p_cov = np.full((1, 1), fill_value=float('+inf'))
    # p_sig = p_cov.diagonal() ** 0.5
    #
    # if verbose:
    #     print('solution: ' + ', '.join(('{:.7E}'.format(p) for p in p_opt)))
    #     print('sigmas: ' + ', '.join(('{:.4f} %'.format(100. * s / abs(p)) for p, s in zip(p_opt, p_sig))))
    #     for p, s in zip(p_opt, p_sig):
    #         print('[{:.7E} --> {:.7E}]'.format(p - 3. * s, p + 3. * s))
    #     show_covariance(p_opt, p_cov, ('mu',), ('m3/s2',), true_values=(mu_true,))
    #
    # mu_opt = p_opt[0]
    # mu_sig = p_sig[0]
    # if (mu_true < mu_opt - 3. * mu_sig) or (mu_true > mu_opt + 3. * mu_sig):
    #     print('!- MU NOT IN 3-SIGMAS -! (mu={:.7E}, {:.2f} sigmas)'.format(mu_true, abs(mu_true - mu_opt) / mu_sig))
    return mu_sig / mu_opt


if __name__ == '__main__':
    bs = np.geomspace(50e3, 3000e3, num=20)
    gammas = np.empty_like(bs)
    for i, b in enumerate(bs):
        gamma = run(mass_true=1e16, vel=10e3, b_sat=b, seed=19960319, verbose=True)
        print('b: {:.1E}, rel_unc: {:.4f} %'.format(b, 100. * gamma))
        gammas[i] = gamma

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bs, y=gammas))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.show()
