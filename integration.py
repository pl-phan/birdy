import numpy as np
import plotly.graph_objects as go
from scipy.integrate import RK45

from flyby_utils import params_to_coords

# scenario parameters
dt = 10.  # s
t_ca = 5. * 3600.  # s
t_max = 10. * 3600.  # s


def gravity(r, mu):
    return - mu * r / np.linalg.norm(r) ** 3.


def variational_eq(r):
    return - r / np.linalg.norm(r) ** 3.


def integrate(mu, vel, b_sat):
    y0 = params_to_coords(vel, b_sat, t_ca)

    dy_dm_0 = np.zeros_like(y0)
    y0 = np.concatenate((y0, dy_dm_0))

    dy = np.empty_like(y0)

    def func(_, y):
        dy[0:3] = y[3:6]
        dy[3:6] = gravity(y[0:3], mu)

        dy[6:9] = y[9:12]
        dy[9:12] = variational_eq(y[0:3])
        return dy

    rk = RK45(func, t0=0., y0=y0, t_bound=t_max, first_step=dt, max_step=dt, rtol=1e-9, atol=1.)

    n_iter = np.ceil(t_max / dt).astype('int')
    ts, dts = np.empty(n_iter, dtype='float'), np.empty(n_iter, dtype='float')
    trajectory = np.empty((n_iter, 6), dtype='float')
    trajectory_var = np.empty((n_iter, 6), dtype='float')

    k = 0
    while rk.status == 'running':
        if k >= n_iter:
            break
        ts[k] = rk.t
        dts[k] = rk.step_size
        trajectory[k] = rk.y[0:6]
        trajectory_var[k] = rk.y[6:12]
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
    M = 1e16  # kg
    v0 = 1500.  # m/s
    b = 100e3  # m

    mu0 = G * M
    t, traj, var = integrate(mu0, v0, b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], name='m={}'.format(M)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=traj[:, 3], y=traj[:, 4], name='m={}'.format(M)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=var[:, 0], y=var[:, 1], name='m={}'.format(M)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=var[:, 3], y=var[:, 4], name='m={}'.format(M)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()
