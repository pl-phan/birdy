import itertools
from collections import namedtuple

import numpy as np
import pandas as pd
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

color_iterator = itertools.cycle(DEFAULT_PLOTLY_COLORS)
TIME_ORIGIN = pd.to_datetime('1858-11-17')
Results = namedtuple('Results', 'x jac')
c = 3e8  # m/s


def random_orthogonal(i, rng=None):
    """
    Returns a 3D vector orthogonal to i
    """
    if not rng:
        rng = np.random.default_rng()
    k = Rotation.random(random_state=rng).apply(np.array((1., 0., 0.)))
    j = np.cross(i, k)
    return j / np.linalg.norm(j)


def mjd2_to_datetime(days, seconds):
    return pd.to_datetime(days + seconds / 86400., unit='D', origin=TIME_ORIGIN)


def datetime_to_mjd2(datetime):
    if isinstance(datetime, str):
        datetime = pd.to_datetime(datetime)
    delta = datetime - TIME_ORIGIN
    return delta.days, delta.seconds


def measurements(probe, observer, f0, win_size=None, delay_noise=None, freq_noise=None, rng=None):
    if not probe.index.equals(observer.index):
        raise ValueError('Indexes of probe and observer mismatch')

    # ranging
    df_relative = probe - observer
    rho = df_relative[['x', 'y', 'z']].apply(np.linalg.norm, axis=1)
    time_delay = 2. * rho / c

    # doppler frequency
    v_r = (df_relative.vx * df_relative.x + df_relative.vy * df_relative.y + df_relative.vz * df_relative.z) / rho
    frequency = f0 * (1. - 2. * v_r / c)

    if win_size:
        time_delay = time_delay.rolling(win_size + (win_size % 2 == 0), center=True).mean().dropna().iloc[::win_size]
        frequency = frequency.rolling(win_size + (win_size % 2 == 0), center=True).mean().dropna().iloc[::win_size]

    if delay_noise or freq_noise:
        if not rng:
            rng = np.random.default_rng()
        if delay_noise:
            time_delay += rng.normal(scale=delay_noise, size=len(time_delay))
        if freq_noise:
            frequency += rng.normal(scale=freq_noise, size=len(frequency))

    return time_delay, frequency


def next_color():
    return next(color_iterator)


def find_sh_coef(harmonics, degree, order, sign):
    if order > degree:
        raise ValueError('order {:d} is larger than degree {:d}'.format(order, degree))
    if sign not in ('c', 's'):
        raise ValueError('sign must be \'c\' or \'s\', but \'{}\' was provided'.format(sign))
    if (order == 0) and (sign != 'c'):
        raise ValueError('sign must be \'c\' when order is 0, but \'{}\' was provided'.format(sign))

    if degree not in harmonics:
        return 0.
    if order not in harmonics[degree]:
        return 0.
    if sign not in harmonics[degree][order]:
        return 0.
    return harmonics[degree][order][sign]


def estim_covariance(residuals, beta0, d_beta):
    m = len(beta0)
    jac = list()
    for b, db, ej in zip(beta0, d_beta, np.eye(m)):
        jac_p = (residuals(beta0 + ej * db) - residuals(beta0)) / db
        jac_m = (residuals(beta0) - residuals(beta0 - ej * db)) / db
        jac.append((jac_p + jac_m) / 2.)
    jac = np.stack(jac, axis=-1)
    return Results(beta0, jac)


def show_covariance(mu, cov, names, units=None, true_values=None, rng=None):
    if not rng:
        rng = np.random.default_rng()
    if not units:
        units = ['' for _ in names]

    n = len(mu)
    samples = rng.multivariate_normal(mean=mu, cov=cov, size=1000000)
    fig = make_subplots(rows=n, cols=n)
    for i in range(n):
        for j in range(n):
            fig.update_xaxes(title_text='{} ({})'.format(names[i], units[i]), col=i + 1, row=j + 1)
            if i == j:
                fig.add_histogram(x=samples[:, i], col=i + 1, row=j + 1,
                                  marker={'color': 'black'}, histnorm='probability density')
            else:
                fig.add_histogram2d(x=samples[:, i], y=samples[:, j], col=i + 1, row=j + 1, coloraxis='coloraxis')
                fig.update_yaxes(title_text='{} ({})'.format(names[j], units[j]), col=i + 1, row=j + 1)
    if true_values:
        for i in range(n):
            for j in range(n):
                if i == j:
                    fig.add_vline(x=true_values[i], col=i + 1, row=j + 1,
                                  line={'color': 'green'})
                else:
                    fig.add_scatter(x=true_values[i:i+1], y=true_values[j:j+1], col=i + 1, row=j + 1,
                                    marker={'color': 'green', 'symbol': 'cross'})
    fig.update_layout(title='Result distribution', showlegend=False, coloraxis=dict(colorscale='greys'))
    fig.update_coloraxes(showscale=False)
    fig.show()


if __name__ == '__main__':
    # TODO TESTS
    show_covariance((100., 5.), ((3., -1.5), (-1.5, 1.)), names=('p1', 'p2'))
