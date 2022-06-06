import numpy as np
from numpy import pi
from scipy.optimize import least_squares

from flyby_utils import params_to_coords
from integration import integrate
from utils import unit_vector, measurements, show_covariance

# physics parameters
G = 6.6743e-11  # m3/kg/s2

# flyby parameters
alpha = 170. * pi / 180.  # rad
beta = -3. * pi / 180.  # rad
delta = 4.5E11  # m
earth = np.array((*unit_vector(alpha, beta), 0., 0., 0.)) * delta

# observation parameters
t_ca = 5. * 3600.  # s
dt = 10.  # s
t_max = 10. * 3600.  # s

# radio parameters
f0 = 8.4e9  # Hz
ranging_noise = 2e-8  # s
doppler_noise = 5e-3  # Hz


def generate_data(mu, sat_init, seed=None):
    t, traj, _ = integrate(mu=mu, y0=sat_init, dt=dt, t_max=t_max)
    return t, measurements(traj, earth, f0, ranging_noise, doppler_noise, seed=seed)


def generate_model(mu, sat_init):
    t, traj, var = integrate(mu=mu, y0=sat_init, dt=dt, t_max=t_max)
    return t, measurements(traj, earth, f0)


def run(mass, vel, b_sat, d_mass=None, seed=None, verbose=0):
    # TODO: choose mode (cubesat), data (data), Jn
    # TODO: plot
    mu = mass * G
    sat_init = params_to_coords(vel, b_sat, t_ca)

    # data generation with noise
    t, (ranging_data, doppler_data) = generate_data(mu, sat_init, seed=seed)
    ranging_uncertainty = ranging_noise
    doppler_uncertainty = doppler_noise

    y = np.concatenate((ranging_data, doppler_data))
    sigmas = np.array((ranging_uncertainty, doppler_uncertainty)).repeat(len(y) / 2)

    def residuals(params):
        if verbose:
            print('Evaluating at {}'.format(params))

        mu_guess, = params
        t2, (ranging_model, doppler_model) = generate_model(mu_guess, sat_init)
        if not np.array_equal(t, t2):
            raise ValueError('time indexes are inconsistent')

        x = np.concatenate((ranging_model, doppler_model))
        return (x - y) / sigmas

    p0 = np.array((mu,))
    dp_rel = np.array((d_mass,)) / mass if d_mass else 1e-3

    results = least_squares(residuals, p0, diff_step=dp_rel)
    p_opt = results.x
    try:
        p_cov = np.linalg.inv(results.jac.T @ results.jac)
    except np.linalg.LinAlgError:
        p_cov = np.full((len(p0), len(p0)), float('+inf'))
    p_sig = p_cov.diagonal() ** 0.5

    if verbose:
        print('solution: ' + ', '.join(('{:.7E}'.format(p) for p in p_opt)))
        print('sigmas: ' + ', '.join(('{:.4f} %'.format(100. * s / abs(p)) for p, s in zip(p_opt, p_sig))))
        for p, s in zip(p_opt, p_sig):
            print('[{:.7E} --> {:.7E}]'.format(p - 3. * s, p + 3. * s))
        show_covariance(p_opt, p_cov, ('mu',), ('m3/s2',), true_values=(mu,))

    mu_opt = p_opt[0]
    mu_sig = p_sig[0]
    if (mu < mu_opt - 3. * mu_sig) or (mu > mu_opt + 3. * mu_sig):
        print('!- MU NOT IN 3-SIGMAS -! (mu={:.7E}, {:.2f} sigmas)'.format(mu, abs(mu - mu_opt) / mu_sig))
    return mu_sig / mu_opt


if __name__ == '__main__':
    gamma = run(mass=1e18, vel=15e3, b_sat=3000e3, seed=19960319, verbose=1)
    print(gamma)
