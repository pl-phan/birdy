import numpy as np
import pandas as pd
from numpy import pi
from plotly.subplots import make_subplots
from scipy.optimize import least_squares

from docks import docks, plot_trajectories
from flyby_utils import shift_pos, params_to_coords, print_flyby
from utils import measurements, next_color, show_covariance

# seed
RNG = np.random.default_rng(19960319)

# physics parameters
G = 6.6743e-11  # m3/kg/s2

# asteroid radius
radius = 49e3  # m

# observation parameters
f0 = 8.4e9  # Hz
t_start = pd.to_datetime('2010-07-10 11:45:00')
t_end = pd.to_datetime('2010-07-10 21:45:00')
dt = 30.
int_time = 60.  # s
delay_noise = 4e-8  # s
freq_noise = 1e-1  # Hz


def generate_data(mass, earth_init, ast_init, v, b_sat, b_cub, alpha, beta, t_ca,
                  ast_harmonics=None, mode='cubesat', verbose=0):
    # flyby parameters
    mu = G * mass  # m3/s2

    # asteroid and Earth propagation
    _, df_earth = docks('earth', t_start, t_end, dt, *np.split(earth_init, 2), verbose=verbose)
    ast_name, df_asteroid = docks('asteroid', t_start, t_end, dt, *np.split(ast_init, 2), verbose=verbose)
    if not df_earth.index.equals(df_asteroid.index):
        raise ValueError('Indexes of earth and asteroid mismatch')

    # find state vectors at close approach time
    t_closest = abs(df_earth.index.to_series() - t_ca).idxmin()
    earth_pos_ca, earth_vel_ca = np.split(df_earth.loc[t_closest].to_numpy(), 2)
    earth_pos_ca = shift_pos(earth_pos_ca, earth_vel_ca, t_from=t_closest, t_to=t_ca)
    ast_pos_ca, ast_vel_ca = np.split(df_asteroid.loc[t_closest].to_numpy(), 2)
    ast_pos_ca = shift_pos(ast_pos_ca, ast_vel_ca, t_from=t_closest, t_to=t_ca)

    sat_pos_ca, sat_vel_ca = params_to_coords(b_sat, v, alpha, beta, ast_pos_ca, ast_vel_ca, earth_pos_ca, rng=RNG)
    cub_pos_ca = (sat_pos_ca - ast_pos_ca) * (b_cub / b_sat) + ast_pos_ca

    # probe and cubesat : find init state vectors and propagate
    sat_pos_init = shift_pos(sat_pos_ca, sat_vel_ca, t_from=t_ca, t_to=t_start)
    cub_pos_init = shift_pos(cub_pos_ca, sat_vel_ca, t_from=t_ca, t_to=t_start)
    _, df_spacecraft = docks('spacecraft', t_start, t_end, dt, sat_pos_init, sat_vel_ca,
                             ast_name, mu, radius, ast_harmonics, verbose=verbose)
    df_cubesat = docks('cubesat', t_start, t_end, dt, cub_pos_init, sat_vel_ca,
                       ast_name, mu, radius, ast_harmonics, verbose=verbose
                       )[1] if (mode == 'cubesat') else None

    if verbose >= 1:
        print_flyby(df_spacecraft, df_asteroid, df_earth)
        if mode == 'cubesat':
            print_flyby(df_cubesat, df_asteroid, df_earth)
        if verbose >= 2:
            plot_trajectories(df_asteroid, df_spacecraft, df_cubesat)

    # time delay and frequency measurements
    if mode == 'earth':
        measuring = (df_spacecraft, df_earth)
    elif mode == 'cubesat':
        measuring = (df_cubesat, df_spacecraft)
    else:
        raise NotImplementedError('available measurement modes : earth or cubesat')

    delay, freq = measurements(
        *measuring, f0=f0, win_size=int(int_time / dt),
        delay_noise=delay_noise, freq_noise=freq_noise, rng=RNG
    )

    sat_init = np.concatenate((sat_pos_init, sat_vel_ca))
    cub_init = np.concatenate((cub_pos_init, sat_vel_ca))
    return (delay, freq), (sat_init, cub_init)


def generate_model(mass, earth_init, ast_init, sat_init, cub_init,
                   ast_harmonics=None, mode='cubesat', verbose=0):
    mu = G * mass  # m3/s2

    # asteroid propagation
    ast_name, df_asteroid = docks('asteroid', t_start, t_end, dt, *np.split(ast_init, 2), verbose=verbose)

    # observer propagation
    if mode == 'earth':
        _, df_observer = docks('earth', t_start, t_end, dt, *np.split(earth_init, 2), verbose=verbose)
        _, df_probe = docks('spacecraft', t_start, t_end, dt, *np.split(sat_init, 2),
                            ast_name, mu, radius, ast_harmonics, verbose=verbose)
    elif mode == 'cubesat':
        _, df_observer = docks('spacecraft', t_start, t_end, dt, *np.split(sat_init, 2),
                               ast_name, mu, radius, ast_harmonics, verbose=verbose)
        _, df_probe = docks('cubesat', t_start, t_end, dt, *np.split(cub_init, 2),
                            ast_name, mu, radius, ast_harmonics, verbose=verbose)
    else:
        raise NotImplementedError('available measurement modes : earth or cubesat')

    # time delay and frequency measurements
    delay, freq = measurements(df_probe, df_observer, f0=f0, win_size=int(int_time / dt))

    return delay, freq


if __name__ == '__main__':
    MODE = 'cubesat'
    VERBOSE = 0

    earth = np.array((4.6E+10, -1.4E+11, 5.3E+06, 2.8E+04, 9.1E+03, -2.5E-01))
    ast = np.array((-4.0E+11, -6.5E+10, 2.1E+10, 4.6E+03, -1.6E+04, -3.8E+02))
    ast_mass = 1.7e18
    ast_sh = {2: {0: {'c': -1.2e-2}, 2: {'c': 2.5e-4, 's': -2.7e-4}}}

    # data generation with noise
    (delay_data, freq_data), (sat, cub) = generate_data(
        mass=ast_mass, earth_init=earth, ast_init=ast, v=1000., b_sat=3000e3, b_cub=100e3,
        alpha=170. * pi / 180., beta=3. * pi / 180., ast_harmonics=ast_sh,
        t_ca=pd.to_datetime('2010-07-10 16:45:00'), mode=MODE, verbose=VERBOSE
    )
    y = np.concatenate((delay_data.to_numpy(), freq_data.to_numpy()))
    delay_uncertainty = delay_noise
    freq_uncertainty = freq_noise
    sigmas = np.array((delay_uncertainty, freq_uncertainty)).repeat(len(y) / 2)


    def residuals(beta):
        mass, c20, c22 = beta
        ast_harmonics = {2: {0: {'c': c20}, 2: {'c': c22, 's': -2.7e-4}}}
        print("Evaluating at {}".format(beta))
        delay_model, freq_model = generate_model(mass, earth, ast, sat, cub, ast_harmonics, mode=MODE, verbose=VERBOSE)
        x = np.concatenate((delay_model.to_numpy(), freq_model.to_numpy()))

        plot_color = next_color()
        fig.add_scatter(x=delay_model.index, y=delay_model, name=str(beta),
                        col=1, row=1, mode='lines', line={'color': plot_color})
        fig.add_scatter(x=freq_model.index, y=freq_model, name=str(beta),
                        col=1, row=2, mode='lines', line={'color': plot_color})
        # return (delay_model.to_numpy() - delay_data.to_numpy()) / delay_uncertainty
        # return (freq_model.to_numpy() - freq_data.to_numpy()) / freq_uncertainty
        return (x - y) / sigmas


    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    color = next_color()
    fig.add_scatter(x=delay_data.index, y=delay_data, name='data', col=1, row=1,
                    mode='markers', marker={'symbol': 'cross', 'color': color})
    fig.add_scatter(x=freq_data.index, y=freq_data, name='data', col=1, row=2,
                    mode='markers', marker={'symbol': 'cross', 'color': color})

    beta0 = np.array((1.5e18, -1e-2, 3e-4))
    results = least_squares(residuals, x0=beta0, diff_step=5e-5, x_scale=(1e18, 1e-2, 1e-4),
                            ftol=1e-6, xtol=5e-5, gtol=float('nan'))
    popt, cov = results.x, np.linalg.inv(results.jac.T @ results.jac)

    p_sigmas = cov.diagonal() ** 0.5

    print('solution: ' + ', '.join(('{:.7E}'.format(p) for p in popt)))
    print('sigmas: ' + ', '.join(('{:.4f} %'.format(100. * s / abs(p)) for p, s in zip(popt, p_sigmas))))
    for p, s in zip(popt, p_sigmas):
        print('[{:.7E} --> {:.7E}]'.format(p - 2. * s, p + 2. * s))
    fig.show()

    show_covariance(popt, cov, ('mass', 'c20', 'c22'), true_values=(1.7e18, -1.2e-2, 2.5e-4))
