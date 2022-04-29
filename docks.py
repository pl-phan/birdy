import filecmp
import os
import subprocess

import pandas as pd
import plotly.graph_objects as go
from ruamel.yaml import YAML

from utils import mjd2_to_datetime, datetime_to_mjd2

LOCAL_DISK = '/local_disk'
DOCKS_DIR = os.path.join(LOCAL_DISK, 'pphan/DOCKS')
DOCKS_ENV = os.path.join(LOCAL_DISK, 'pphan/envs/docks/bin/python')


def docks(name, t_start, t_end, dt, init_pos, init_vel,
          asteroid_name=None, asteroid_mu=None, verbose=0):

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
    work_dir = os.path.join(DOCKS_DIR, 'bodies', '{}_{}'.format(name, pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')))

    # Initial conditions file
    init_txt = '{:d}    {:.3f}    {:.17E}    {:.17E}    {:.17E}    {:.17E}    {:.17E}    {:.17E}\n'.format(
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
    config['timeSettings']['propagation_time'][1] = '{:d}:{:d}:{:.3f}'.format(
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


def plot_trajectories(df_asteroid, df_spacecraft, df_cubesat):
    figure = go.Figure(layout={'scene': {'aspectmode': 'data'}})
    figure.add_scatter3d(
        x=df_asteroid.x, y=df_asteroid.y, z=df_asteroid.z, mode='markers', name='asteroid',
        marker={'size': 1, 'color': df_asteroid.index.astype('int'), 'colorscale': 'viridis'}
    )
    figure.add_scatter3d(
        x=df_spacecraft.x, y=df_spacecraft.y, z=df_spacecraft.z, mode='markers', name='spacecraft',
        marker={'size': 1, 'color': df_spacecraft.index.astype('int'), 'colorscale': 'viridis'}
    )
    figure.add_scatter3d(
        x=df_cubesat.x, y=df_cubesat.y, z=df_cubesat.z, mode='markers', name='cubesat',
        marker={'size': 1, 'color': df_cubesat.index.astype('int'), 'colorscale': 'viridis'}
    )
    figure.show()


if __name__ == '__main__':
    # TODO TESTS
    pass
