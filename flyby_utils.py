import numpy as np
import pandas as pd
from numpy import pi
from scipy.spatial.transform import Rotation

from utils import random_orthogonal


def shift_pos(pos, vel, delta_t=None, t_from=None, t_to=None):
    """
    Shifts position along a linear trajectory
    """
    # Check inputs
    if not (delta_t or (t_from and t_to)):
        raise ValueError('Either delta_t or the time limits must be specified')
    if not delta_t:
        delta_t = (t_to - t_from) / pd.to_timedelta(1, 's')
    # Update position
    new_pos = pos + vel * delta_t
    return new_pos


def close_approach_calculator(pos, vel, ast_pos, ast_vel):
    """
    Calculates position at CA, and time of CA
    """
    # Center on asteroid
    pos_rel = pos - ast_pos
    vel_rel = vel - ast_vel

    # Analytic solution of CA
    delta_t = - pos_rel @ vel_rel / np.linalg.norm(vel_rel) ** 2.

    # Update positions
    new_pos = shift_pos(pos, vel, delta_t=delta_t)
    new_ast_pos = shift_pos(ast_pos, ast_vel, delta_t=delta_t)
    return new_pos, new_ast_pos, delta_t


def params_to_coords(b, v, alpha, beta, ast_pos, ast_vel, obs, rng=None):
    """
    Calculates CA coordinates from flyby parameters (b, v, alpha, beta)
    """
    # Center on asteroid
    obs_rel = obs - ast_pos

    # Initial coordinates
    pos_rel = np.array((0., -b, 0.))
    vel_rel = np.array((v, 0., 0.))

    # Create rotation pointing away from obs_rel with Euler angles (alpha, beta)
    pivot_inv = Rotation.from_euler('ZY', np.array((-alpha, -beta))).as_matrix()

    # Create random rotation back to inertial reference
    i = obs_rel / np.linalg.norm(obs_rel)
    j = random_orthogonal(i, rng=rng)
    k = np.cross(i, j)
    pointing_inv = np.array((i, j, k)).T

    # Combine rotations
    rot_inv = pointing_inv @ pivot_inv
    pos = rot_inv @ pos_rel + ast_pos
    vel = rot_inv @ vel_rel + ast_vel

    return pos, vel


def coords_to_params(pos, vel, ast_pos, ast_vel, obs):
    """
    Calculates flyby parameters (b, v, alpha, beta) from initial coordinates
    """
    # Position at CA
    pos_ca, ast_pos_ca, _ = close_approach_calculator(pos, vel, ast_pos, ast_vel)

    # Center on asteroid
    obs_rel = obs - ast_pos_ca
    pos_rel = pos_ca - ast_pos_ca
    vel_rel = vel - ast_vel

    # Calculate flyby asteroid parameters (b, v)
    b = np.linalg.norm(pos_rel)
    v = np.linalg.norm(vel_rel)

    # Calculate flyby observer parameters (alpha, beta)
    # Create new vector base
    i = vel_rel / v
    j = - pos_rel / b
    k = np.cross(i, j)
    # Project observer into new base
    rot = np.linalg.inv(np.array((i, j, k)).T)
    obs_rot = rot @ obs_rel
    alpha = np.arctan2(obs_rot[1], obs_rot[0])
    beta = np.arctan2(obs_rot[2], np.linalg.norm(obs_rot[0:2]))

    return v, b, alpha, beta


def print_flyby(probe, asteroid, observer):
    # find close approach
    t_closest = (probe - asteroid)[['x', 'y', 'z']].apply(np.linalg.norm, axis=1).idxmin()

    obs_pos_ca, obs_vel_ca = np.split(observer.loc[t_closest].to_numpy(), 2)
    ast_pos_ca, ast_vel_ca = np.split(asteroid.loc[t_closest].to_numpy(), 2)
    sat_pos_ca, sat_vel_ca = np.split(probe.loc[t_closest].to_numpy(), 2)

    sat_pos_ca, ast_pos_ca, delta_t = close_approach_calculator(sat_pos_ca, sat_vel_ca, ast_pos_ca, ast_vel_ca)
    obs_pos_ca = shift_pos(obs_pos_ca, obs_vel_ca, delta_t=delta_t)
    t_ca = t_closest + pd.to_timedelta(delta_t, 's')

    # true flyby parameters
    v, b, alpha, beta = coords_to_params(sat_pos_ca, sat_vel_ca, ast_pos_ca, ast_vel_ca, obs_pos_ca)
    print("t_ca={}, v={:.3f} m/s, b={:.3f} km, alpha={:.3f} deg, beta={:.3f} deg".format(
        t_ca, v, b / 1e3, alpha * 180. / pi, beta * 180. / pi
    ))


if __name__ == '__main__':
    # TODO TESTS
    pass
