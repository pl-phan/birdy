import os.path
from zlib import adler32

import numpy as np
import plotly.graph_objects as go
from numpy import pi
from scipy.integrate import RK45
from scipy.spatial.transform import Rotation
from scipy.special import legendre

from flyby_utils import params_to_coords

# scenario parameters
dt = 60.  # s
t_ca = 4. * 3600.  # s
t_max = 10. * 3600.  # s


def integrate(gravity_field, r_ast, vel, b, bz=0., alpha=pi / 2., beta=0.):
    config = np.array((*gravity_field, r_ast, alpha, beta, vel, b, bz, dt, t_ca, t_max))
    filename = './data/{}.npz'.format(adler32(config.tobytes()))

    if os.path.isfile(filename):
        data = np.load(filename, allow_pickle=False)
        return data['t'], data['traj'], data['traj_var']

    y0 = params_to_coords(vel, b, bz, t_ca)
    rot = Rotation.from_euler('XY', (beta, alpha - pi / 2.))
    y0 = rot.apply(y0.reshape(2, 3)).reshape(6)

    jac_shape = (6, len(gravity_field))
    y_var_0 = np.zeros(jac_shape)

    dy = np.empty_like(y0)
    dy_var = np.empty_like(y_var_0)

    def func(y, y_var):
        dy_var[0:3, :] = y_var[3:6, :]
        # variational equations for the gravity field
        rho = np.linalg.norm(y[0:3])
        sin_phi = y[2] / rho
        alpha = r_ast / rho
        mu4 = gravity_field[0] / rho ** 4.

        # classical gravitational force
        dy_var[3:6, 0] = - y[0:3] / rho ** 3.

        # zonal gravitational force
        beta = alpha
        for n in range(2, dy_var.shape[1] + 1):
            beta *= alpha
            c1 = (n + 1) * legendre(n)(sin_phi) * rho
            c2 = legendre(n).deriv()(sin_phi)
            dy_var[3:5, n - 1] = mu4 * beta * y[0:2] * (c1 + c2 * y[2])
            dy_var[5, n - 1] = mu4 * beta * (c1 * y[2] - c2 * y[0:2] @ y[0:2])

        # 2nd law of motion
        dy[0:3] = y[3:6]
        dy[3:6] = dy_var[3:6, :] @ gravity_field
        return dy, dy_var

    y_vec_0 = np.concatenate((y0, y_var_0.flatten()))

    def wrapper(_, y_vec):
        _dy, _dy_var = func(y_vec[:6], y_vec[6:].reshape(jac_shape))
        return np.concatenate((_dy, _dy_var.flatten()))

    rk = RK45(wrapper, t0=0., y0=y_vec_0, t_bound=t_max,
              first_step=dt, max_step=dt, rtol=1e-9, atol=1.)

    n_iter = np.ceil(t_max / dt).astype('int')
    ts, dts = np.empty(n_iter, dtype='float'), np.empty(n_iter, dtype='float')
    trajectory = np.empty((n_iter, 6), dtype='float')
    trajectory_var = np.empty((n_iter, *jac_shape), dtype='float')

    i = 0
    while rk.status == 'running':
        if i >= n_iter:
            break
        ts[i] = rk.t
        dts[i] = rk.step_size

        trajectory[i] = rk.y[:6]
        trajectory_var[i] = rk.y[6:].reshape(jac_shape)
        rk.step()
        i += 1

    if rk.status == 'failed':
        raise ValueError('failed integration')
    # check constant step size
    dts = dts[1:]
    if min(dts) < dt:
        raise ValueError('step_size went down to {}'.format(min(dts)))

    np.savez(filename, t=ts, traj=trajectory, traj_var=trajectory_var)
    return ts, trajectory, trajectory_var


if __name__ == '__main__':
    G = 6.6743e-11  # m3/kg/s2
    grav = np.array((G * 1.7e18, 1.2e-2, 4.1e-5, 5.7e-4))
    t, traj, traj_var = integrate(grav, 50e3, 0. * pi / 180., 15. * pi / 180., 1e3, 100e3)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=traj[:, 0], name='x'))
    fig.add_trace(go.Scatter(x=t, y=traj[:, 1], name='y'))
    fig.add_trace(go.Scatter(x=t, y=traj[:, 2], name='z'))
    fig.update_layout(title='r')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=traj[:, 3], name='vx'))
    fig.add_trace(go.Scatter(x=t, y=traj[:, 4], name='vy'))
    fig.add_trace(go.Scatter(x=t, y=traj[:, 5], name='vz'))
    fig.update_layout(title='v')
    fig.show()

    gravity_coefs = ['mu'] + ['J{:d}'.format(i) for i, _ in enumerate(grav[1:], 2)]
    for j, coef in enumerate(gravity_coefs):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=traj_var[:, 0, j], name='x'))
        fig.add_trace(go.Scatter(x=t, y=traj_var[:, 1, j], name='y'))
        fig.add_trace(go.Scatter(x=t, y=traj_var[:, 2, j], name='z'))
        fig.update_layout(title='dr/d' + coef)
        fig.show()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=traj_var[:, 3, j], name='vx'))
        fig.add_trace(go.Scatter(x=t, y=traj_var[:, 4, j], name='vy'))
        fig.add_trace(go.Scatter(x=t, y=traj_var[:, 5, j], name='vz'))
        fig.update_layout(title='dv/d' + coef)
        fig.show()
