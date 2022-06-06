import numpy as np
from numpy import pi


def shift_pos(pos, vel, delta_t):
    return pos + vel * delta_t


def close_approach_calculator(pos, vel):
    # Analytic solution of CA
    delta_t = - pos @ vel / np.linalg.norm(vel) ** 2.
    # Update positions
    new_pos = shift_pos(pos, vel, delta_t)
    return new_pos, delta_t


def coords_to_params(pos, vel, obs):
    # Calculate flyby asteroid parameters (b, v)
    v = np.linalg.norm(vel)
    b = np.linalg.norm(pos)

    # Calculate flyby observer parameters (alpha, beta)
    # Create new vector base
    i = vel / v
    j = - pos / b
    k = np.cross(i, j)
    # Project observer into new base
    rot = np.linalg.inv(np.array((i, j, k)).T)
    obs_rot = rot @ obs
    alpha = np.arctan2(obs_rot[1], obs_rot[0])
    beta = np.arctan2(obs_rot[2], np.linalg.norm(obs_rot[0:2]))

    return v, b, alpha, beta


def params_to_coords(v, b, time_close_approach):
    x0 = -v * time_close_approach
    return np.array((x0, -b, 0., v, 0., 0.))


def print_flyby_params(t, trajectory, observer):
    # find close approach
    i_ca = np.linalg.norm(trajectory[:, 0:3], axis=1).argmin()
    t_ca = t[i_ca]
    pos_ca = trajectory[i_ca, 0:3]
    vel_ca = trajectory[i_ca, 3:6]

    pos_ca, delta_t = close_approach_calculator(pos_ca, vel_ca)
    t_ca = t_ca + delta_t

    # true flyby parameters
    v, b, alpha, beta = coords_to_params(pos_ca, vel_ca, observer)
    print("t_ca={:.1f}, v={:.1f} m/s, b={:.3f} km, alpha={:.3f} deg, beta={:.3f} deg".format(
        t_ca, v, b / 1e3, alpha * 180. / pi, beta * 180. / pi
    ))


if __name__ == '__main__':
    # TODO TESTS
    pass
