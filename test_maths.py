from builtins import enumerate

import numpy as np
import plotly.graph_objects as go
from scipy.integrate import RK45

from scipy.optimize import curve_fit


# Parameters

# physics parameters
G = 6.6743e-11  # m3/kg/s2

# asteroid parameters
R = 49e3  # m
m0 = 1.7e18  # kg

# flyby parameters
# v = 50  # m/s
# b = R * 20.  # m


def propagate(ts, mass):
    print('Called with m = {}'.format(mass))
    mu = G * mass  # m3/s2

    def y_dot(_, y):
        """trajectory propagation due to asteroid gravity"""
        dy = np.empty_like(y)
        dy[0:2] = y[2:4]
        dy[2:4] = - mu * y[0:2] / np.linalg.norm(y[0:2]) ** 3
        # dy[4:6] = dy[2:4] / mass
        return dy

    # time boundaries
    t_start = 0. * 3600.  # s
    t_end = 40. * 3600.  # s

    # initial state vector
    # probe position
    p_x = 1000e3  # m
    p_y = -3000e3  # m
    # probe velocity
    p_vx = -10.  # m/s
    p_vy = 50.  # m/s
    # state vector
    y0 = np.array([p_x, p_y, p_vx, p_vy], dtype='float')
    # # partial derivative
    # d_err_x = 0.
    # d_err_y = 0.
    # y0 = np.array([p_x, p_y, p_vx, p_vy, d_err_x, d_err_y], dtype='float')

    # propagate trajectory
    dt = 60.  # s
    integrator = RK45(y_dot, t_start, y0, t_end, dt, rtol=1e-9, atol=1e-12, first_step=dt)

    # collect data
    ts, ys, dts = list(), list(), list()
    while integrator.status == 'running':
        integrator.step()
        ts.append(integrator.t)
        ys.append(integrator.y)
        dts.append(integrator.step_size)
    ts = np.array(ts[:-1])
    pos, vel = np.split(np.stack(ys[:-1]), 2, axis=1)
    # pos, vel, d_err = np.split(np.stack(ys[:-1]), 3, axis=1)

    # verify constant step size
    if len(set(dts)) > 1:
        print(max(dts), min(dts))
        raise ValueError('DT too long')

    return np.concatenate((vel[:, 0], vel[:, 1]))


# measurements
D = propagate(None, m0)
noise = 5.  # m/s
D += np.random.normal(scale=noise, size=D.shape)
# # plot measurements
# fig1 = go.Figure()
# fig1.update_yaxes(scaleanchor='x', scaleratio=1)
# fig1.add_scatter(x=D[:, 0], y=D[:, 1], mode='markers', name='probe velocity')
# fig1.show()

# masses = np.geomspace(0.5 * m0, 2. * m0, num=21)
# s = np.empty_like(masses)
# mass_est = np.empty_like(masses)
# for i, m in enumerate(masses):
#     c, err = propagate(m)
#     s[i] = np.linalg.norm(c - D) ** 2.
#     mass_est[i] = m - err.flatten() @ (c - D).flatten() / np.linalg.norm(err) ** 2.
#
# fig3 = go.Figure()
# fig3.update_yaxes(scaleanchor='x', scaleratio=1)
# fig3.add_scatter(x=masses, y=mass_est, mode='markers', name='estimate')
# fig3.show()

p_opt, p_cov = curve_fit(propagate, None, D, p0=2. * m0)
sigma = p_cov.diagonal() ** 0.5
print('{}, {}%'.format(p_opt, 100. * sigma / p_opt))
