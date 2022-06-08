import numpy as np
from utils import mul_1d, dot_1d

c = 3e8  # m/s
f0 = 8.4e9  # Hz


def measure(trajectory, trajectory_var):
    # ranging
    rho = dot_1d(trajectory[:, 0:3], trajectory[:, 0:3]) ** 0.5
    ranging = rho * 2. / c
    # doppler
    vr = dot_1d(trajectory[:, 0:3], trajectory[:, 3:6]) / rho
    doppler = - vr * 2. * f0 / c

    # ranging variations
    rho_var = dot_1d(trajectory[:, 0:3], trajectory_var[:, 0:3, :])
    rho_var = mul_1d(rho_var, 1. / rho)
    ranging_var = rho_var * 2. / c
    # doppler variations
    vr_var = (dot_1d(trajectory[:, 0:3], trajectory_var[:, 3:6, :])
              + dot_1d(trajectory[:, 3:6], trajectory_var[:, 0:3, :])
              + mul_1d(vr, rho_var))
    vr_var = mul_1d(vr_var, 1. / rho)
    doppler_var = - vr_var * 2. * f0 / c

    return (np.concatenate((ranging, doppler), axis=0),
            np.concatenate((ranging_var, doppler_var), axis=0))


def add_noise(measurements, ranging_noise, doppler_noise, seed=None):
    rng = np.random.default_rng(seed)
    sigmas = np.array((ranging_noise, doppler_noise)).repeat(len(measurements) / 2)
    return measurements + rng.normal(scale=sigmas), sigmas


if __name__ == '__main__':
    # TODO TESTS
    pass
