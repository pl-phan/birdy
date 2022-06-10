import numpy as np
from numpy import pi
from scipy.spatial.transform import Rotation

from integration import integrate
from measurement import measure, add_noise
from optimization import Optimizer
from utils import show_covariance, show_fit

# physics parameters
G = 6.6743e-11  # m3/kg/s2

# observer parameters
alpha_obs = 170. * pi / 180.  # rad
beta_obs = -3. * pi / 180.  # rad
delta_obs = 4.5e11  # m

# measurement  noise
ranging_noise = 5e-8  # s
doppler_noise = 1e-2  # Hz


def get_relative_uncertainty(gravity_field, r_ast, alpha_ast, beta_ast,
                             vel, b_sat, seed=None, verbose=0):
    # TODO: choose mode (cubesat), data (ranging/doppler)

    earth = np.array((delta_obs, 0., 0.))
    earth = Rotation.from_euler('ZY', (alpha_obs, beta_obs)).apply(earth)
    earth = Rotation.from_euler('XY', (beta_ast, alpha_ast - pi / 2.)).apply(earth)
    earth = np.concatenate((earth, np.zeros_like(earth)))

    def model(params):
        if verbose >= 2:
            print('called at :', params)
        t, traj, traj_var = integrate(params, r_ast, alpha_ast, beta_ast, vel, b_sat)
        y, y_var = measure(traj - earth, traj_var)
        return t, y, y_var

    # theoretical model
    t_data, y_model, _ = model(gravity_field)
    # data with noise
    y_data, sigmas = add_noise(y_model, ranging_noise, doppler_noise, seed=seed)

    # least squares fit
    p0 = gravity_field
    opt = Optimizer(model, y_data, sigmas)
    p_opt, p_cov, p_rel = opt.least_squares(p0, method='no_fit')

    # print results
    msg = 'true gravitational field: GM={:.7E}'.format(gravity_field[0])
    for i, j in enumerate(gravity_field[1:], 2):
        msg += ' J{:d}={:.7E}'.format(i, j)
    msg += '\n'
    print(msg)
    msg = 'found gravitational field:\nGM={:.7E} ± {:.5f}%'.format(p_opt[0], 100. * p_rel[0])
    for i, (j, r) in enumerate(zip(p_opt[1:], p_rel[1:]), 2):
        msg += '\nJ{:d}={:.7E} ± {:.5f}%'.format(i, j, 100. * r)
    msg += '\n'
    print(msg)

    if verbose >= 1:
        # show distribution
        gravity_coefs = ['mu'] + ['J{:d}'.format(i) for i, _ in enumerate(p_rel[1:], 2)]
        gravity_units = ['m3/s2'] + ['' for _ in p_rel[1:]]
        show_covariance(p_opt, p_cov, gravity_coefs, gravity_units,
                        true_values=gravity_field, n_samples=1000000, seed=seed)

    if verbose >= 2:
        # show data fit
        show_fit(t_data, y_data, sigmas, model, p_opt, p_cov, n_samples=5, seed=seed)

    return p_rel


if __name__ == '__main__':
    rng = np.random.default_rng(19960319)
    for s in rng.uniform(111111, 999999, size=5).astype('int'):
        sig = get_relative_uncertainty(
            # gravity_field=np.array((G * 1.7e18, 1.2e-2, 4.1e-5, 5.7e-4)),
            gravity_field=np.array((G * 1.7e18, 1.2e-2)),
            r_ast=50e3, alpha_ast=0. * pi / 180., beta_ast=15. * pi / 180.,
            vel=1e3, b_sat=100e3,
            seed=s, verbose=2
        )
