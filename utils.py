import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


TIME_ORIGIN = pd.to_datetime('1858-11-17')


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
