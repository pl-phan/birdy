import numpy as np

c = 3e8  # m/s
f0 = 8.4e9  # Hz


def dot_1d(a, b):
    return np.einsum('ij,ij->i', a, b)


def measure(trajectory, trajectory_var):
    # ranging
    rho = np.linalg.norm(trajectory[:, 0:3], axis=1)
    ranging = 2. * rho / c
    # doppler
    vr = dot_1d(trajectory[:, 0:3], trajectory[:, 3:6]) / rho
    doppler = - 2. * f0 * vr / c

    # ranging
    rho_var = dot_1d(trajectory[:, 0:3], trajectory_var[:, 0:3]) / rho
    ranging_var = 2. * rho_var / c
    # doppler
    vr_var = (dot_1d(trajectory[:, 3:6], trajectory_var[:, 0:3])
              + dot_1d(trajectory[:, 0:3], trajectory_var[:, 3:6])
              + vr @ rho_var
              ) / rho
    doppler_var = - 2. * f0 * vr_var / c

    return np.concatenate((ranging, doppler)), np.concatenate((ranging_var, doppler_var))


def add_noise(measurements, ranging_noise, doppler_noise, seed=None):
    rng = np.random.default_rng(seed)
    sigmas = np.array((ranging_noise, doppler_noise)).repeat(len(measurements) / 2)
    return measurements + rng.normal(scale=sigmas), sigmas


if __name__ == '__main__':
    # TODO TESTS
    pass
