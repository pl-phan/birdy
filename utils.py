import itertools

import numpy as np
import pandas as pd
from plotly.colors import DEFAULT_PLOTLY_COLORS
from scipy.spatial.transform import Rotation

color_iterator = itertools.cycle(DEFAULT_PLOTLY_COLORS)
TIME_ORIGIN = pd.to_datetime('1858-11-17')
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


if __name__ == '__main__':
    # TODO TESTS
    pass
