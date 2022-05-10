import filecmp
import os
import subprocess

import pandas as pd
import plotly.graph_objects as go
from ruamel.yaml import YAML

from utils import mjd2_to_datetime, datetime_to_mjd2, find_sh_coef

LOCAL_DISK = '/local_disk'
DOCKS_DIR = os.path.join(LOCAL_DISK, 'pphan/DOCKS')
DOCKS_ENV = os.path.join(LOCAL_DISK, 'pphan/envs/docks/bin/python')


def docks(name, t_start, t_end, dt, init_pos, init_vel,
          ast_name=None, ast_mu=None, ast_radius=None, ast_harmonics=None, verbose=0):

    test_files = ('init.txt', 'config.yaml', 'ast_harmonics.tab')
    test_dirs = [os.path.join(DOCKS_DIR, 'bodies', name)
                 for name in os.listdir(os.path.join(DOCKS_DIR, 'bodies'))
                 if name.startswith(name)]
    work_dir = config_writer(name, t_start, t_end, dt, init_pos, init_vel, ast_name, ast_mu, ast_radius, ast_harmonics)

    for test_dir in test_dirs:
        # Compare files, see https://docs.python.org/3/library/filecmp.html#filecmp.cmpfiles
        if len(filecmp.cmpfiles(test_dir, work_dir, test_files)[0]) == len(test_files):
            if not os.path.isfile(os.path.join(test_dir, 'traj.txt')):
                continue
            # Trajectory already exists
            for test_file in test_files:
                os.remove(os.path.join(work_dir, test_file))
            os.removedirs(work_dir)
            if verbose:
                print('Using already existing dir {}'.format(test_dir))
            return os.path.basename(test_dir), docks_parser(os.path.join(test_dir, 'traj.txt'))

    # Else create new trajectory
    if verbose:
        print('Creating new dir {}'.format(work_dir))

    subprocess.run(
        (DOCKS_ENV, os.path.join(DOCKS_DIR, 'Propagator/propagator.py'), os.path.join(work_dir, 'config.yaml')),
        stdout=subprocess.DEVNULL if verbose < 1 else None, stderr=subprocess.STDOUT
    )
    return os.path.basename(work_dir), docks_parser(os.path.join(work_dir, 'traj.txt'))


def config_writer(name, t_start, t_end, dt, init_pos, init_vel,
                  ast_name=None, ast_mu=None, ast_radius=None, ast_harmonics=None):
    work_dir = os.path.join(DOCKS_DIR, 'bodies', '{}_{}'.format(name, pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')))
    os.makedirs(work_dir, exist_ok=True)

    # spherical harmonics file
    harmonics_txt = ''
    if ast_harmonics:
        deg_max = max(ast_harmonics)
        harmonics_txt += '{:.17E}, {:.17E}, 0.0, {:d}, {:d}, 1, 0.0, 0.0\n'.format(
            ast_radius / 1e3, ast_mu / 1e9, deg_max, deg_max)

        for l_deg in range(1, deg_max + 1):
            harmonics_txt += '{:d}, 0, {:.17E}, 0.0\n'.format(l_deg, find_sh_coef(ast_harmonics, l_deg, 0, 'c'))
            for m_ord in range(1, l_deg + 1):
                harmonics_txt += '{:d}, {:d}, {:.17E}, {:.17E}\n'.format(
                    l_deg, m_ord,
                    find_sh_coef(ast_harmonics, l_deg, m_ord, 'c'),
                    find_sh_coef(ast_harmonics, l_deg, m_ord, 's')
                )
    with open(os.path.join(work_dir, 'ast_harmonics.tab'), 'w') as f:
        f.write(harmonics_txt)

    # Initial conditions file
    init_txt = '{:d}    {:.3f}    {:.17E}    {:.17E}    {:.17E}    {:.17E}    {:.17E}    {:.17E}\n'.format(
        *datetime_to_mjd2(t_start), *(init_pos / 1e3), *(init_vel / 1e3)
    )
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

    if ast_name:
        if ast_harmonics:
            config['perturbations']['complex_grav_model_activated'] = True
            config['complex_grav_bodies']['body1']['name'] = ast_name
            config['complex_grav_bodies']['body1']['ephFile'] = os.path.join('../', ast_name, 'traj.txt')
            config['complex_grav_bodies']['body1']['rotMatrixFile'] = '../lutetia_rotation_matrix.txt'
            config['complex_grav_bodies']['body1']['sphCoeffFile'] = 'ast_harmonics.tab'
            config['complex_grav_bodies']['body1']['sphHarmDegree'] = max(ast_harmonics)
        else:
            config['perturbations']['new_bodies_added'] = True
            config['new_grav_bodies']['body1']['name'] = ast_name
            config['new_grav_bodies']['body1']['mu'] = '{:.17E}'.format(ast_mu)
            config['new_grav_bodies']['body1']['ephFile'] = os.path.join('../', ast_name, 'traj.txt')

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
