import numpy as np
import plotly.graph_objects as go
from scipy.integrate import RK45


# def gravity(r, r0, mu):
#     delta_r = r - r0
#     return - mu * delta_r / np.linalg.norm(delta_r) ** 3.


def variational_eq(r, r0):
    delta_r = r - r0
    return - delta_r / np.linalg.norm(delta_r) ** 3.


def integrate(mu, y0, dt, t_max):
    dy_dm_0 = np.zeros_like(y0)
    y0 = np.concatenate((y0, dy_dm_0))

    r0 = np.array((0., 0., 0.))
    dy = np.empty_like(y0)

    def func(_, y):
        dy[6:9] = y[9:12]
        dy[9:12] = variational_eq(y[0:3], r0)

        dy[0:3] = y[3:6]
        dy[3:6] = dy[9:12] * mu
        # dy[3:6] = gravity(y[0:3], r0, mu)
        return dy

    rk = RK45(func, t0=0., y0=y0, t_bound=t_max, first_step=dt, max_step=dt, rtol=1e-9, atol=1.)

    n_iter = np.ceil(t_max / dt).astype('int')
    ts, dts = np.empty(n_iter, dtype='float'), np.empty(n_iter, dtype='float')
    trajectory, variations = np.empty((n_iter, 6), dtype='float'), np.empty((n_iter, 6), dtype='float')

    k = 0
    while rk.status == 'running':
        if k >= n_iter:
            break
        ts[k] = rk.t
        dts[k] = rk.step_size
        trajectory[k] = rk.y[0:6]
        variations[k] = rk.y[6:12]
        rk.step()
        k += 1

    if rk.status == 'failed':
        raise ValueError('failed integration')
    dts = dts[1:]
    if min(dts) < dt:
        raise ValueError('step_size went down to {}'.format(min(dts)))

    return ts, trajectory, variations


if __name__ == '__main__':
    G = 6.6743e-11  # m3/kg/s2
    M = 1e16  # kg
    v0 = 1500.  # m/s
    b = 100e3  # m
    t_tot = 4. * 3600.

    mu0 = G * M
    t, traj, var = integrate(mu=mu0, y0=np.array((-v0 * t_tot / 2., -b, 0., v0, 0., 0.)), dt=10., t_max=t_tot)

    d_M = M / 10000.  # m3/s2
    d_mu = G * d_M
    _, traj1, _ = integrate(mu=mu0 - d_mu, y0=np.array((-v0 * t_tot / 2., -b, 0., v0, 0., 0.)), dt=10., t_max=t_tot)
    _, traj2, _ = integrate(mu=mu0 + d_mu, y0=np.array((-v0 * t_tot / 2., -b, 0., v0, 0., 0.)), dt=10., t_max=t_tot)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=traj1[:, 0], y=traj1[:, 1], name='m={}'.format(M + d_M)))
    fig.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], name='m={}'.format(M)))
    fig.add_trace(go.Scatter(x=traj2[:, 0], y=traj2[:, 1], name='m={}'.format(M + d_M)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=traj1[:, 3], y=traj1[:, 4], name='m={}'.format(M + d_M)))
    fig.add_trace(go.Scatter(x=traj[:, 3], y=traj[:, 4], name='m={}'.format(M)))
    fig.add_trace(go.Scatter(x=traj2[:, 3], y=traj2[:, 4], name='m={}'.format(M + d_M)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()

    for i, s in enumerate(('x', 'y', 'z', 'vx', 'vy', 'vz')):
        if 'z' in s:
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=var[:, i], name='{} var_eq'.format(s)))
        fig.add_trace(go.Scatter(x=t, y=(traj2[:, i] - traj1[:, i]) / d_M, name='{} diff'.format(s)))
        fig.add_trace(go.Scatter(x=t, y=var[:, i] - (traj2[:, i] - traj1[:, i]) / d_M, name='{} error'.format(s)))
        fig.show()
