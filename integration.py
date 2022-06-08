import numpy as np
import plotly.graph_objects as go
from scipy.integrate import RK45

from flyby_utils import params_to_coords

# scenario parameters
dt = 10.  # s
t_ca = 4. * 3600.  # s
t_max = 10. * 3600.  # s


def gravity(r, gravity_field):
    return - gravity_field[0] * r / np.linalg.norm(r) ** 3.


def variational_eq(r, degree):
    if degree == 1:
        degree = 0

    if degree == 0:
        return - r / np.linalg.norm(r) ** 3.
    else:
        raise NotImplementedError


def integrate(gravity_field, vel, b_sat):
    jac_shape = (6, len(gravity_field))
    y0 = params_to_coords(vel, b_sat, t_ca)
    y_var_0 = np.zeros(jac_shape)

    dy = np.empty_like(y0)
    dy_var = np.empty_like(y_var_0)

    def func(y, y_var):
        dy[0:3] = y[3:6]
        dy[3:6] = gravity(y[0:3], gravity_field)
        for i in range(len(gravity_field)):
            dy_var[0:3, i] = y_var[3:6, i]
            dy_var[3:6, i] = variational_eq(y[0:3], i + 1)
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

    k = 0
    while rk.status == 'running':
        if k >= n_iter:
            break
        ts[k] = rk.t
        dts[k] = rk.step_size

        trajectory[k] = rk.y[:6]
        trajectory_var[k] = rk.y[6:].reshape(jac_shape)
        rk.step()
        k += 1

    if rk.status == 'failed':
        raise ValueError('failed integration')
    dts = dts[1:]
    if min(dts) < dt:
        raise ValueError('step_size went down to {}'.format(min(dts)))

    return ts, trajectory, trajectory_var


if __name__ == '__main__':
    G = 6.6743e-11  # m3/kg/s2
    grav = np.array((G * 1.7e18, 1.9e-2, -1.2e-3, -6.5e-3))
    v = 15e3  # m/s
    b = 3000e3  # m

    t, traj, traj_var = integrate(grav, v, b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=traj[:, 0], name='x'))
    fig.add_trace(go.Scatter(x=t, y=traj[:, 1], name='y'))
    fig.update_layout(title='r')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=traj[:, 3], name='vx'))
    fig.add_trace(go.Scatter(x=t, y=traj[:, 4], name='vy'))
    fig.update_layout(title='v')
    fig.show()

    gravity_coefs = ['mu'] + ['J{:d}'.format(i) for i, _ in enumerate(grav[1:], 2)]
    for j, coef in enumerate(gravity_coefs):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=traj_var[:, 0, j], name='x'))
        fig.add_trace(go.Scatter(x=t, y=traj_var[:, 1, j], name='y'))
        fig.update_layout(title='dr/d' + coef)
        fig.show()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=traj_var[:, 3, j], name='vx'))
        fig.add_trace(go.Scatter(x=t, y=traj_var[:, 4, j], name='vy'))
        fig.update_layout(title='dv/d' + coef)
        fig.show()
