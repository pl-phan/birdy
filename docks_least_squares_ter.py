import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import pi
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

from docks import docks, relative_time_index, plot_trajectory
from flyby_utils import shift_pos, close_approach_calculator, params_to_coords, coords_to_params
from utils import measurements, next_color, normalize

# physics parameters
G = 6.6743e-11  # m3/kg/s2
c = 3e8  # m/s

# observation parameters
f0 = 8.4e9  # Hz
t_start = pd.to_datetime('2010-07-10 11:45:00')
t_end = pd.to_datetime('2010-07-10 21:45:00')
dt = 30.
delay_noise = 2e-9  # s
freq_noise = 2e-3  # Hz


def generate_data(mass, obs_init, ast_init, v, b_sat, alpha, beta, t_ca, verbose=False):
    # flyby parameters
    mu = G * mass  # m3/s2

    # propagation
    _, df_observer = docks('earth', t_start, t_end, dt, *np.split(obs_init, 2))
    t_closest = abs(df_observer.index.to_series() - t_ca).idxmin()
    obs_pos_ca, obs_vel_ca = np.split(df_observer.loc[t_closest].to_numpy(), 2)
    obs_pos_ca = shift_pos(obs_pos_ca, obs_vel_ca, t_from=t_closest, t_to=t_ca)

    ast_name, df_asteroid = docks('asteroid', t_start, t_end, dt, *np.split(ast_init, 2))
    t_closest = abs(df_asteroid.index.to_series() - t_ca).idxmin()
    ast_pos_ca, ast_vel_ca = np.split(df_asteroid.loc[t_closest].to_numpy(), 2)
    ast_pos_ca = shift_pos(ast_pos_ca, ast_vel_ca, t_from=t_closest, t_to=t_ca)

    sat_pos_ca, sat_vel_ca = params_to_coords(b_sat, v, alpha, beta, ast_pos_ca, ast_vel_ca, obs_pos_ca)
    sat_pos_init = shift_pos(sat_pos_ca, sat_vel_ca, t_from=t_ca, t_to=t_start)
    _, df_spacecraft = docks('spacecraft', t_start, t_end, dt, sat_pos_init, sat_vel_ca, ast_name, mu)

    if verbose:
        # Find real CA
        t_closest = (df_spacecraft - df_asteroid)[['x', 'y', 'z']].apply(np.linalg.norm, axis=1).idxmin()
        obs_pos_ca, obs_vel_ca = np.split(df_observer.loc[t_closest].to_numpy(), 2)
        ast_pos_ca, ast_vel_ca = np.split(df_asteroid.loc[t_closest].to_numpy(), 2)
        sat_pos_ca, sat_vel_ca = np.split(df_spacecraft.loc[t_closest].to_numpy(), 2)
        sat_pos_ca, ast_pos_ca, delta_t = close_approach_calculator(sat_pos_ca, sat_vel_ca, ast_pos_ca, ast_vel_ca)
        obs_pos_ca = shift_pos(obs_pos_ca, obs_vel_ca, delta_t=delta_t)
        t_ca = t_closest + pd.to_timedelta(delta_t, 's')

        b_, v_, alpha_, beta_ = coords_to_params(sat_pos_ca, sat_vel_ca, ast_pos_ca, ast_vel_ca, obs_pos_ca)
        print(t_ca, b_, v_, alpha_ * 180. / pi, beta_ * 180. / pi)

    df_observer = relative_time_index(df_observer, t_start)
    df_asteroid = relative_time_index(df_asteroid, t_start)
    df_spacecraft = relative_time_index(df_spacecraft, t_start)

    if verbose:
        # plot trajectory
        fig0 = go.Figure(layout={'scene': {'aspectmode': 'data'}})
        plot_trajectory(df_asteroid, 'asteroid', fig0)
        plot_trajectory(df_spacecraft, 'spacecraft', fig0)
        fig0.show()

    # measurements
    delay_earth, freq_earth = measurements(
        df_spacecraft, df_observer, f0=f0,
        delay_noise=delay_noise, freq_noise=freq_noise
    )
    sat_init = np.concatenate((sat_pos_init, sat_vel_ca))
    return (delay_earth, freq_earth), sat_init


def generate_model(mass, obs_init, ast_init, sat_init):
    mu = G * mass  # m3/s2

    # propagation
    _, df_observer = docks('earth', t_start, t_end, dt, *np.split(obs_init, 2))
    ast_name, df_asteroid = docks('asteroid', t_start, t_end, dt, *np.split(ast_init, 2))
    _, df_spacecraft = docks('spacecraft', t_start, t_end, dt, *np.split(sat_init, 2),
                             asteroid_name=ast_name, asteroid_mu=mu)

    df_observer = relative_time_index(df_observer, t_start)
    df_spacecraft = relative_time_index(df_spacecraft, t_start)

    # measurements
    delay_earth, freq_earth = measurements(df_spacecraft, df_observer, f0=f0)

    return delay_earth, freq_earth


if __name__ == '__main__':
    obs = np.array((4.6E+10, -1.4E+11, 5.3E+06, 2.8E+04, 9.1E+03, -2.5E-01))
    ast = np.array((-4.0E+11, -6.5E+10, 2.1E+10, 4.6E+03, -1.6E+04, -3.8E+02))

    (delay_data, freq_data), sat = generate_data(
        mass=1.7e18, obs_init=obs, ast_init=ast,
        v=15e3, b_sat=3000e3, alpha=170. * pi / 180., beta=3. * pi / 180.,
        t_ca=pd.to_datetime('2010-07-10 15:46:04'), verbose=False
    )

    delay_ref, freq_ref = generate_model(0., obs, ast, sat)
    y_delay = (delay_data - delay_ref).to_numpy()
    minmax_delay = min(y_delay), max(y_delay)
    y_freq = (freq_data - freq_ref).to_numpy()
    minmax_freq = min(y_freq), max(y_freq)
    y = np.concatenate((normalize(y_delay, *minmax_delay), normalize(y_freq, *minmax_freq)))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    color = next_color()
    fig.add_scatter(x=delay_ref.index, y=y_delay, name='data',
                    col=1, row=1, mode='markers', marker={'symbol': 'cross', 'color': color})
    fig.add_scatter(x=freq_ref.index, y=y_freq, name='data', showlegend=False,
                    col=1, row=2, mode='markers', marker={'symbol': 'cross', 'color': color})


    def wrapper(_, mass):
        print('Called with m = {}'.format(mass))
        delay_model, freq_model = generate_model(float(mass), obs, ast, sat)
        x_delay = (delay_model - delay_ref).to_numpy()
        x_freq = (freq_model - freq_ref).to_numpy()
        x = np.concatenate((normalize(x_delay, *minmax_delay), normalize(x_freq, *minmax_freq)))
        plot_color = next_color()
        fig.add_scatter(x=delay_ref.index, y=x_delay, name='mass {}'.format(mass),
                        col=1, row=1, mode='lines', line={'color': plot_color})
        fig.add_scatter(x=freq_ref.index, y=x_freq, name='mass {}'.format(mass), showlegend=False,
                        col=1, row=2, mode='lines', line={'color': plot_color})
        return x

    p_opt, p_cov = curve_fit(wrapper, xdata=None, ydata=y, p0=1.4e18, epsfcn=1e-9)
    sigma = p_cov.diagonal() ** 0.5
    print('{}, {}%'.format(p_opt, 100. * sigma / p_opt))
    print('[{} -> {}]'.format(p_opt - 2. * sigma, p_opt + 2. * sigma))

    fig.show()
