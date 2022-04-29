import filecmp
import os
import subprocess

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ruamel.yaml import YAML

from flyby_utils import close_approach_calculator
from utils import mjd2_to_datetime, datetime_to_mjd2

LOCAL_DISK = '/local_disk'
DOCKS_DIR = os.path.join(LOCAL_DISK, 'pphan/DOCKS')
DOCKS_ENV = os.path.join(LOCAL_DISK, 'pphan/envs/docks/bin/python')


def docks(name, t_start, t_end, dt, init_pos, init_vel,
          asteroid_name=None, asteroid_mu=None, verbose=0):
    if isinstance(t_start, str):
        t_start = pd.to_datetime(t_start)
    if isinstance(t_end, str):
        t_end = pd.to_datetime(t_end)

    test_dirs = [os.path.join(DOCKS_DIR, 'bodies', name)
                 for name in os.listdir(os.path.join(DOCKS_DIR, 'bodies'))
                 if name.startswith(name)]
    work_dir = config_writer(name, t_start, t_end, dt, init_pos, init_vel, asteroid_name, asteroid_mu)

    for test_dir in test_dirs:
        # Compare files, see https://docs.python.org/3/library/filecmp.html#filecmp.cmpfiles
        if len(filecmp.cmpfiles(test_dir, work_dir, ('init.txt', 'config.yaml'))[0]) == 2:
            if not os.path.isfile(os.path.join(test_dir, 'traj.txt')):
                continue
            # Trajectory already exists
            os.remove(os.path.join(work_dir, 'init.txt'))
            os.remove(os.path.join(work_dir, 'config.yaml'))
            os.removedirs(work_dir)
            if verbose >= 1:
                print('Using already existing dir {}'.format(test_dir))
            return os.path.basename(test_dir), docks_parser(os.path.join(test_dir, 'traj.txt'))

    # Else create new trajectory
    if verbose >= 1:
        print('Creating new dir {}'.format(work_dir))

    subprocess.run(
        (DOCKS_ENV, os.path.join(DOCKS_DIR, 'Propagator/propagator.py'), os.path.join(work_dir, 'config.yaml')),
        stdout=subprocess.DEVNULL if verbose < 1 else None, stderr=subprocess.STDOUT
    )
    return os.path.basename(work_dir), docks_parser(os.path.join(work_dir, 'traj.txt'))


def config_writer(name, t_start, t_end, dt, init_pos, init_vel, asteroid_name=None, asteroid_mu=None):
    work_dir = os.path.join(DOCKS_DIR, 'bodies', '{}_{}'.format(name, pd.Timestamp.now().strftime('%Y%m%d%H%M%S')))

    # Initial conditions file
    init_txt = '{:d}    {:d}    {:.17E}    {:.17E}    {:.17E}    {:.17E}    {:.17E}    {:.17E}\n'.format(
        *datetime_to_mjd2(t_start), *(init_pos / 1e3), *(init_vel / 1e3)
    )
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, 'init.txt'), 'w') as f:
        f.write(init_txt)

    # Configuration file
    yaml = YAML()
    with open('default_config.yaml', 'r') as f:
        config = yaml.load(f)
    duration = (t_end - t_start).components
    config['timeSettings']['propagation_time'][0] = float(duration.days)
    config['timeSettings']['propagation_time'][1] = '{:d}:{:d}:{:.1f}'.format(
        duration.hours, duration.minutes, duration.seconds
    )
    config['timeSettings']['time_step'][0] = dt
    if asteroid_name is not None:
        config['perturbations']['new_bodies_added'] = True
        config['new_grav_bodies']['body1']['name'] = asteroid_name
        config['new_grav_bodies']['body1']['mu'] = '{:.17E}'.format(asteroid_mu)
        config['new_grav_bodies']['body1']['ephFile'] = os.path.join('../', asteroid_name, 'traj.txt')
    with open(os.path.join(work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    return work_dir


def docks_parser(filename, convert_to_meters=True):
    skip_rows = 2
    with open(filename, 'r') as f:
        while f.readline() != 'META_STOP\n':
            skip_rows += 1

    df = pd.read_csv(
        filename, header=None, skiprows=skip_rows, sep='\t',
        names=['days', 'seconds', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
    )
    df['time'] = mjd2_to_datetime(df.days, df.seconds)
    df = df[['time', 'x', 'y', 'z', 'vx', 'vy', 'vz']].set_index('time')

    if convert_to_meters:
        df *= 1e3

    return df


def relative_time_index(df, center_timestamp):
    df.index = (df.index - center_timestamp) / pd.to_timedelta(1, 's')
    return df


def plot_trajectory(df, name, figure, backend='plotly'):
    if backend == 'plotly':
        figure.add_scatter3d(
            x=df.x, y=df.y, z=df.z, mode='markers', name=name,
            marker={'size': 1, 'color': df.index, 'colorscale': 'viridis'}
        )
    # elif backend == 'matplotlib':
    #     figure.plot(df.x, df.y, '.-', label=name)
    else:
        raise NotImplementedError('{} unknown'.format(backend))


if __name__ == '__main__':
    # Import
    df_lutetia = docks_parser('./DOCKS/backup/lutetia/traj.txt')
    df_rosetta = docks_parser('./DOCKS/backup/rosetta/traj.txt')
    df_cubesat = docks_parser('./DOCKS/backup/cubesat/traj.txt')

    # Center on cubesat CA
    t0 = df_lutetia.index[0]
    lut_pos = df_lutetia[['x', 'y', 'z']].iloc[0].to_numpy()
    lut_vel = df_lutetia[['vx', 'vy', 'vz']].mean().to_numpy()
    cub_pos = df_cubesat[['x', 'y', 'z']].iloc[0].to_numpy()
    cub_vel = df_cubesat[['vx', 'vy', 'vz']].mean().to_numpy()
    _, _, t_ca = close_approach_calculator(cub_pos, cub_vel, lut_pos, lut_vel)
    t_ca = t0 + pd.to_timedelta(t_ca, 's')
    df_lutetia = relative_time_index(df_lutetia, t_ca)
    df_rosetta = relative_time_index(df_rosetta, t_ca)
    df_cubesat = relative_time_index(df_cubesat, t_ca)

    # Plot trajectories
    fig = go.Figure(layout={'scene': {'aspectmode': 'data'}})
    plot_trajectory(df_lutetia, name='21 Lutetia', figure=fig)
    plot_trajectory(df_rosetta, name='Rosetta', figure=fig)
    plot_trajectory(df_cubesat, name='cubesat', figure=fig)
    fig.show()

    # Plot distance
    df_rosetta['dist'] = np.linalg.norm((df_rosetta[['x', 'y', 'z']] - df_lutetia[['x', 'y', 'z']]), axis=-1)
    df_cubesat['dist'] = np.linalg.norm((df_cubesat[['x', 'y', 'z']] - df_lutetia[['x', 'y', 'z']]), axis=-1)
    fig = go.Figure()
    fig.add_scatter(x=df_rosetta.index, y=df_rosetta.dist)
    fig.add_scatter(x=df_cubesat.index, y=df_cubesat.dist)
    fig.show()
