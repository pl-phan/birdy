import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


TIME_ORIGIN = pd.to_datetime('1858-11-17')
c = 3e8  # m/s


def orthogonal(i, random=False):
    """
    Returns a 3D vector orthogonal to i
    """
    k = np.array((1., 0., 0.))
    if random:
        k = Rotation.random().apply(k)

    j = np.cross(i, k)
    return j / np.linalg.norm(j)


def mjd2_to_datetime(days, seconds):
    return pd.to_datetime(days + seconds / 86400., unit='D', origin=TIME_ORIGIN)


def datetime_to_mjd2(datetime):
    if isinstance(datetime, str):
        datetime = pd.to_datetime(datetime)
    delta = datetime - TIME_ORIGIN
    return delta.days, delta.seconds


def measurements(probe, observer, f0, win_size=None):
    if not probe.index.equals(observer.index):
        raise ValueError('Indexes of probe and observer mismatch')

    # ranging
    df_relative = probe - observer
    rho = df_relative[['x', 'y', 'z']].apply(np.linalg.norm, axis=1)
    time_delay = 2. * rho / c

    # doppler frequency
    v_r = (df_relative.vx * df_relative.x + df_relative.vy * df_relative.y + df_relative.vz * df_relative.z) / rho
    frequency = f0 * (1. - 2. * v_r / c)

    if win_size is not None:
        time_delay = time_delay.rolling(win_size + (win_size % 2 == 0), center=True).mean().dropna().iloc[::win_size]
        frequency = frequency.rolling(win_size + (win_size % 2 == 0), center=True).mean().dropna().iloc[::win_size]

    return time_delay, frequency
