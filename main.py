import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import pi
from plotly.subplots import make_subplots
from scipy.optimize import least_squares

from docks import docks, plot_trajectories
from flyby_utils import shift_pos, params_to_coords, print_flyby
from utils import measurements, next_color, estim_covariance, show_covariance

# physics parameters
G = 6.6743e-11  # m3/kg/s2

# observation parameters
f0 = 8.4e9  # Hz
t_start = pd.to_datetime('2010-07-10 11:45:00')
t_end = pd.to_datetime('2010-07-10 21:45:00')
dt = 30.
int_time = 60.  # s
delay_noise = 2e-8  # s
freq_noise = 5e-3  # Hz


def generate_data(mass, earth_init, ast_init, v, b_sat, b_cub, alpha, beta, t_ca,
                  ast_radius=None, ast_harmonics=None, mode='cubesat', seed=None, verbose=0):
    # Seed
    rng = np.random.default_rng(seed)

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

    sat_pos_ca, sat_vel_ca = params_to_coords(b_sat, v, alpha, beta, ast_pos_ca, ast_vel_ca, earth_pos_ca, rng=rng)
    cub_pos_ca = (sat_pos_ca - ast_pos_ca) * (b_cub / b_sat) + ast_pos_ca

    # probe and cubesat : find init state vectors and propagate
    sat_pos_init = shift_pos(sat_pos_ca, sat_vel_ca, t_from=t_ca, t_to=t_start)
    cub_pos_init = shift_pos(cub_pos_ca, sat_vel_ca, t_from=t_ca, t_to=t_start)
    _, df_spacecraft = docks('spacecraft', t_start, t_end, dt, sat_pos_init, sat_vel_ca,
                             ast_name, mu, ast_radius, ast_harmonics, verbose=verbose)
    df_cubesat = docks('cubesat', t_start, t_end, dt, cub_pos_init, sat_vel_ca,
                       ast_name, mu, ast_radius, ast_harmonics, verbose=verbose
                       )[1] if (mode == 'cubesat') else None

    if verbose:
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
        delay_noise=delay_noise, freq_noise=freq_noise, rng=rng
    )

    sat_init = np.concatenate((sat_pos_init, sat_vel_ca))
    cub_init = np.concatenate((cub_pos_init, sat_vel_ca))
    return (delay, freq), (sat_init, cub_init)


def generate_model(mass, earth_init, ast_init, sat_init, cub_init,
                   ast_radius=None, ast_harmonics=None, mode='cubesat', verbose=0):
    mu = G * mass  # m3/s2

    # asteroid propagation
    ast_name, df_asteroid = docks('asteroid', t_start, t_end, dt, *np.split(ast_init, 2), verbose=verbose)

    # observer propagation
    if mode == 'earth':
        _, df_observer = docks('earth', t_start, t_end, dt, *np.split(earth_init, 2), verbose=verbose)
        _, df_probe = docks('spacecraft', t_start, t_end, dt, *np.split(sat_init, 2),
                            ast_name, mu, ast_radius, ast_harmonics, verbose=verbose)
    elif mode == 'cubesat':
        _, df_observer = docks('spacecraft', t_start, t_end, dt, *np.split(sat_init, 2),
                               ast_name, mu, ast_radius, ast_harmonics, verbose=verbose)
        _, df_probe = docks('cubesat', t_start, t_end, dt, *np.split(cub_init, 2),
                            ast_name, mu, ast_radius, ast_harmonics, verbose=verbose)
    else:
        raise NotImplementedError('available measurement modes : earth or cubesat')

    # time delay and frequency measurements
    delay, freq = measurements(df_probe, df_observer, f0=f0, win_size=int(int_time / dt))

    return delay, freq


def run(mode, data, mass, b_sat, b_cub, vel, seed=None, verbose=0, method='estim', d_mass=None):
    earth = np.array((4.6E+10, -1.4E+11, 5.3E+06, 2.8E+04, 9.1E+03, -2.5E-01))
    ast = np.array((-4.0E+11, -6.5E+10, 2.1E+10, 4.6E+03, -1.6E+04, -3.8E+02))

    # data generation with noise
    (delay_data, freq_data), (sat, cub) = generate_data(
        mass=mass, earth_init=earth, ast_init=ast, v=vel, b_sat=b_sat, b_cub=b_cub,
        alpha=170. * pi / 180., beta=3. * pi / 180.,
        t_ca=pd.to_datetime('2010-07-10 15:45:00'), mode=mode, seed=seed, verbose=verbose
    )
    delay_uncertainty = delay_noise
    freq_uncertainty = freq_noise

    fig0 = None
    if verbose:
        fig0 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
        color = next_color()
        fig0.add_scatter(x=delay_data.index, y=delay_data, name='data', col=1, row=1,
                         mode='markers', marker={'symbol': 'cross', 'color': color})
        fig0.add_scatter(x=freq_data.index, y=freq_data, name='data', col=1, row=2,
                         mode='markers', marker={'symbol': 'cross', 'color': color})

    if data == 'ranging':
        y = delay_data.to_numpy()
        sigmas = np.full_like(y, delay_uncertainty)
    elif data == 'doppler':
        y = freq_data.to_numpy()
        sigmas = np.full_like(y, freq_uncertainty)
    elif data == 'both':
        y = np.concatenate((delay_data.to_numpy(), freq_data.to_numpy()))
        sigmas = np.array((delay_uncertainty, freq_uncertainty)).repeat(len(y) / 2)
    else:
        raise NotImplementedError

    def residuals(beta):
        mass_guess, = beta
        if verbose:
            print('Evaluating at {}'.format(beta))
        delay_model, freq_model = generate_model(mass_guess, earth, ast, sat, cub, mode=mode, verbose=verbose)

        if verbose:
            plot_color = next_color()
            fig0.add_scatter(x=delay_model.index, y=delay_model, name=str(beta),
                             col=1, row=1, mode='lines', line={'color': plot_color})
            fig0.add_scatter(x=freq_model.index, y=freq_model, name=str(beta),
                             col=1, row=2, mode='lines', line={'color': plot_color})

        if data == 'ranging':
            x = delay_model.to_numpy()
        elif data == 'doppler':
            x = freq_model.to_numpy()
        elif data == 'both':
            x = np.concatenate((delay_model.to_numpy(), freq_model.to_numpy()))
        else:
            raise NotImplementedError

        return (x - y) / sigmas

    beta0 = np.array((mass,))
    if d_mass is None:
        d_beta = beta0 * 1e-3
    else:
        d_beta = np.array((d_mass,))
    if method == 'estim':
        results = estim_covariance(residuals, beta0, d_beta)
    elif method == 'least_squares':
        results = least_squares(residuals, x0=beta0, diff_step=5e-5, ftol=1e-6, xtol=5e-5, gtol=float('nan'))
    else:
        raise NotImplementedError

    p_opt = results.x
    try:
        p_cov = np.linalg.inv(results.jac.T @ results.jac)
        # # noinspection PyTupleAssignmentBalance
        # _, s, VT = svd(results.jac, full_matrices=False)
        # threshold = np.finfo(float).eps * max(results.jac.shape) * s[0]
        # s = s[s > threshold]
        # VT = VT[:s.size]
        # p_cov = np.dot(VT.T / s**2, VT)
    except np.linalg.LinAlgError:
        p_cov = np.full((len(beta0), len(beta0)), float('+inf'))
    p_sigmas = p_cov.diagonal() ** 0.5

    if verbose:
        print('solution: ' + ', '.join(('{:.7E}'.format(p) for p in p_opt)))
        print('sigmas: ' + ', '.join(('{:.4f} %'.format(100. * s / abs(p)) for p, s in zip(p_opt, p_sigmas))))
        for p, s in zip(p_opt, p_sigmas):
            print('[{:.7E} --> {:.7E}]'.format(p - 3. * s, p + 3. * s))
        fig0.show()
        show_covariance(p_opt, p_cov, ('mass',), ('kg',), true_values=(mass,))

    mass_optim = p_opt[0]
    mass_sigma = p_sigmas[0]
    if (mass < mass_optim - 3. * mass_sigma) or (mass > mass_optim + 3. * mass_sigma):
        print('!- MASS NOT IN 3-SIGMAS -! (m={:.7E}, {:.2f} sigmas)'.format(mass, abs(mass - mass_optim) / mass_sigma))
    return mass_sigma / mass_optim


def run2(mode, masses, b_sat, b_cub, vel, seed=None, verbose=0):
    earth = np.array((4.6E+10, -1.4E+11, 5.3E+06, 2.8E+04, 9.1E+03, -2.5E-01))
    ast = np.array((-4.0E+11, -6.5E+10, 2.1E+10, 4.6E+03, -1.6E+04, -3.8E+02))

    # data generation with noise
    _, (sat, cub) = generate_data(
        mass=masses[0], earth_init=earth, ast_init=ast, v=vel, b_sat=b_sat, b_cub=b_cub,
        alpha=170. * pi / 180., beta=3. * pi / 180.,
        t_ca=pd.to_datetime('2010-07-10 15:45:00'), mode=mode, seed=seed, verbose=verbose
    )
    delay_ref, freq_ref = generate_model(0., earth, ast, sat, cub, mode=mode, verbose=verbose)

    fig0 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    for m in masses:
        delay_model, freq_model = generate_model(m, earth, ast, sat, cub, mode=mode, verbose=verbose)

        plot_color = next_color()
        fig0.add_scatter(x=delay_model.index, y=delay_model - delay_ref, name='m={:.2E}'.format(m),
                         col=1, row=1, mode='lines', line={'color': plot_color})
        fig0.add_scatter(x=freq_model.index, y=freq_model - freq_ref, name='m={:.2E}'.format(m),
                         col=1, row=2, mode='lines', line={'color': plot_color}, showlegend=False)
    fig0.show()


if __name__ == '__main__':
    seed_meta = 19960319
    # run2('earth', np.linspace(1e18, 2e18, num=5), 3000, -3000, 1500, seed=seed_meta, verbose=2)
    rng_meta = np.random.default_rng(seed_meta)
    (b_sat, b_cub, mass) = 3000e3, 1000e3, 3e16
    res_file = 'results_{:d}_{:d}_{:d}_{:d}.csv'.format(int(b_sat), int(b_cub), int(mass), seed_meta)

    DATA = 'doppler'

    logs = list()
    for vel in np.geomspace(100, 10000, num=11):
        print('dist: {:.7E}'.format(vel))
        SEED = int(rng_meta.uniform(111111, 999999))

        sigma1 = run('earth', DATA, mass, b_sat, -b_cub, vel, seed=SEED, verbose=0)
        print('\t\tmode: earth --> {:.3f}%'.format(100. * sigma1))

        sigma2 = run('cubesat', DATA, mass, b_sat, -b_cub, vel, seed=SEED, verbose=0)
        print('\t\tmode: cubesat --> {:.3f}%'.format(100. * sigma2))

        logs.append(('cubesat', DATA, vel, sigma1 / sigma2))

        df = pd.DataFrame(data=logs, columns=('mode', 'data', 'vel', 'alpha'))
        df.to_csv(res_file, index=False)

    df = pd.read_csv(res_file)
    df = df.sort_values(['mode', 'data', 'vel'])
    df.to_csv(res_file, index=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.vel, y=df.alpha))
    fig.update_xaxes(type='log')
    fig.update_yaxes(type='log')
    fig.show()
