import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import pi
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from integration import integrate
from measurement import measure, add_noise
from optimization import Optimizer
from utils import mul_1d, jac2cov, show_covariance, show_fit

# physics parameters
G = 6.6743e-11  # m3/kg/s2

# observer parameters
alpha_obs = 170. * pi / 180.  # rad
beta_obs = -3. * pi / 180.  # rad
delta_obs = 4.5e11  # m

# measurement  noise
ranging_noise = 5e-8  # s
doppler_noise = 1e-2  # Hz


def get_relative_uncertainty(gravity_field, r_ast, vel, b_sat, b_cub=None, b_cub_z=None,
                             alpha_ast=pi / 2., beta_ast=0., method='no_fit', seed=None, verbose=0):
    # TODO visualize 3D

    earth = np.array((delta_obs, 0., 0.))
    earth = Rotation.from_euler('ZY', (alpha_obs, beta_obs)).apply(earth)
    earth = Rotation.from_euler('XY', (beta_ast, alpha_ast - pi / 2.)).apply(earth)
    earth = np.concatenate((earth, np.zeros_like(earth)))

    def model(params):
        if verbose >= 2:
            print('called at :', params)

        t, traj_sat, traj_sat_var = integrate(params, r_ast, vel, b_sat, alpha=alpha_ast, beta=beta_ast)
        if b_cub:
            _, traj_cub, traj_cub_var = integrate(params, r_ast, vel, b_cub, bz=b_cub_z, alpha=alpha_ast, beta=beta_ast)
            y, y_var = measure(traj_cub - traj_sat, traj_cub_var - traj_sat_var)
        else:
            y, y_var = measure(traj_sat - earth, traj_sat_var)

        return t, y, y_var

    # theoretical model
    t_data, y_model, y_model_var = model(gravity_field)
    # data with noise
    y_data, sigmas = add_noise(y_model, ranging_noise, doppler_noise, seed=seed)

    if method == 'no_fit':
        p_opt, p_cov, p_rel = jac2cov(gravity_field, mul_1d(y_model_var, 1. / sigmas))
    else:
        # least squares fit
        opt = Optimizer(model, y_data, sigmas)
        p_opt, p_cov, p_rel = opt.least_squares(gravity_field, method=method)

    if verbose >= 1:
        # print results
        msg = '\ntrue gravitational field: GM={:.7E}'.format(gravity_field[0])
        for i, j in enumerate(gravity_field[1:], 2):
            msg += ' J{:d}={:.7E}'.format(i, j)
        print(msg)
        msg = 'found gravitational field:\nGM={:.7E} ± {:.5f}%'.format(p_opt[0], 100. * p_rel[0])
        for i, (j, r) in enumerate(zip(p_opt[1:], p_rel[1:]), 2):
            msg += '\nJ{:d}={:.7E} ± {:.5f}%'.format(i, j, 100. * r)
        msg += '\n'
        print(msg)

    if verbose >= 2:
        # show distribution
        gravity_coefs = ['mu'] + ['J{:d}'.format(i) for i, _ in enumerate(p_rel[1:], 2)]
        gravity_units = ['m3/s2'] + ['' for _ in p_rel[1:]]
        show_covariance(p_opt, p_cov, gravity_coefs, gravity_units,
                        true_values=gravity_field, n_samples=100000, seed=seed)

    if verbose >= 3:
        # show data fit
        show_fit(t_data, y_data, sigmas, model, p_opt, p_cov, n_samples=5, seed=seed)

    return p_rel


if __name__ == '__main__':
    rng = np.random.default_rng(19960319)
    s = int(rng.uniform(11111, 99999))

    vs = np.geomspace(10., 15e3, num=50)
    bs = np.geomspace(100e3, 3000e3, num=50)

    x0 = 1e16
    log_x0 = np.array((np.log(x0),))
    target = 5e-2
    log_target = np.log(target)

    def find_min_mass(vel, b_sat, verbose=0):

        def wrapper(params):
            mass, = np.exp(params)
            sig = get_relative_uncertainty(gravity_field=np.array((G * mass,)),
                                           r_ast=50e3, vel=vel, b_sat=b_sat, seed=s, verbose=verbose)
            log_sig = np.log(sig[0])
            if verbose >= 1:
                print('mass={:.7E} kg ; sig={:.5f} %'.format(mass, 100. * sig[0]))
                print('log_mass={:.7E} ; log_sig={:.5f}'.format(params[0], log_sig))
            return (log_sig - log_target) ** 2.

        res = minimize(wrapper, x0=log_x0, method='Nelder-Mead', options={'disp': True})
        res = np.exp(res.x[0])

        print('\t\t v={:.0f}, b={:.0f}, mass={:.2E}'.format(vel, b_sat, res))
        return res

    ms = np.empty((len(bs), len(vs)))
    ms_table = list()
    for j, v in enumerate(vs):
        for i, b in enumerate(bs):
            ms[i, j] = find_min_mass(v, b)
            ms_table.append((v, b, ms[i, j]))
    ms_table = pd.DataFrame(ms_table, columns=('v', 'b', 'm_5p'))
    print(ms_table)
    ms_table.to_csv('v_b_m5p.csv', index=False)

    fig = go.Figure(data=go.Contour(x=vs, y=bs, z=np.log10(ms)))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.show()
