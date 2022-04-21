import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import pi
from scipy.spatial.transform import Rotation

from utils import orthogonal


def shift_pos(pos, vel, delta_t=None, t_from=None, t_to=None):
    """
    Shifts position along a linear trajectory
    """
    # Check inputs
    if delta_t is None and (t_from is None or t_to is None):
        raise ValueError('Either delta_t or the time limits must be specified')
    if delta_t is None:
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

    return b, v, alpha, beta


def params_to_coords(b, v, alpha, beta, ast_pos, ast_vel, obs):
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
    j = orthogonal(i)
    k = np.cross(i, j)
    pointing_inv = np.array((i, j, k)).T

    # Combine rotations
    rot_inv = pointing_inv @ pivot_inv
    pos = rot_inv @ pos_rel + ast_pos
    vel = rot_inv @ vel_rel + ast_vel

    return pos, vel


if __name__ == '__main__':

    earth = np.array((4.673357097257601e+07, -1.442547290674476e+08, 5.344020919850666e+03))  # km
    lut_pos = np.array((-4.016720013405068e+08, -6.505189545820294e+07, 2.064459008439349e+07))  # km
    lut_vel = np.array((4.612019734177062e+00, -1.634318247843924e+01, -3.820462659452210e-01))  # km/s
    
    ros_pos = np.array((-4.016718885570552e+08, -6.504870292705259e+07, 2.064398450989245e+07))  # km
    ros_vel = np.array((-1.037512395457134e+01, -1.590181263960469e+01, -1.516779165699269e-01))  # km/s
    
    ros_pos, lut_pos, _ = close_approach_calculator(ros_pos, ros_vel, lut_pos, lut_vel)

    b_param, v_param, alpha_param, beta_param = coords_to_params(ros_pos, ros_vel, lut_pos, lut_vel, earth)
    print("b: {:.1f} km, v: {:.4f} km/s, alpha: {:.2f} deg, beta: {:.2f} deg".format(
        b_param, v_param, alpha_param * 180. / pi, beta_param * 180. / pi))

    # b_param = 3168.
    # v_param = 14.99
    # alpha_param = 172.18 * pi / 180.
    # beta_param = -3. * pi / 180.

    fig = go.Figure(layout={'scene': {'aspectmode': 'data'}})

    def plot(pos, vel, color, size, name=None, with_line=True):
        fig.add_scatter3d(x=(pos - lut_pos)[0:1], y=(pos - lut_pos)[1:2], z=(pos - lut_pos)[2:3],
                          mode='markers', marker={'color': color, 'size': size},
                          name=name, showlegend=(name is not None))
        if not with_line:
            return
        line = np.stack(((pos - lut_pos), (pos - lut_pos) + 500. * (vel - lut_vel)))
        fig.add_scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2],
                          mode='lines', line={'color': color, 'width': size},
                          showlegend=False)

    for _ in range(200):
        ros_, ros_vel_ = params_to_coords(b_param, v_param, alpha_param, beta_param, lut_pos, lut_vel, earth)
        plot(ros_, ros_vel_, color='blue', size=1.)
    plot(ros_pos, ros_vel, color='red', size=3., name='Rosetta')

    plot(lut_pos, lut_vel, color='black', size=3., name='21 Lutetia', with_line=False)
    obs_line = np.stack((lut_pos * 0., (earth - lut_pos) * 1e-6))
    fig.add_scatter3d(x=obs_line[:, 0], y=obs_line[:, 1], z=obs_line[:, 2],
                      mode='lines', line={'color': 'black', 'width': 3.},
                      showlegend=False)
    fig.show()
