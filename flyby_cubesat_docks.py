import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import pi

from docks import docks, center_time_index, plot_trajectory
from flyby_utils import shift_pos, close_approach_calculator, params_to_coords, coords_to_params

# # # # # # # # # # # # # # # #
# Simulation

# Parameters

# physics parameters
G = 6.6743e-11  # m3/kg/s2
c = 3e8  # m/s

# # asteroid parameters
R = 49e3  # m
M = 1.7e16  # kg
GM = G * M  # m3/s2

# # flyby parameters
v = 14.99e3  # m/s
b_sat = 3000e3  # m
b_cub = -R * 10.  # m
alpha = 172.18 * pi / 180.  # rad
beta = 3.0 * pi / 180.  # rad
t_ca = pd.to_datetime('2010-07-10 15:46:04')

# # observation parameters
f0 = 8.4e9  # Hz
t0 = pd.to_datetime('2010-07-10 11:45:00')
t1 = pd.to_datetime('2010-07-10 21:45:00')
dt = 10.
int_time = 60.  # s
freq_noise = 4e-3  # Hz
tof_noise = 5e-1  # s

# Propagation
obs_pos_init = np.array((4.63302E+10, -1.44387E+11, 5.34725E+06))
obs_vel_init = np.array((2.78472E+04, 9.07371E+03, -2.51332E-01))
_, df_observer = docks('earth', t0, t1, dt, obs_pos_init, obs_vel_init)
t_closest = abs(df_observer.index.to_series() - t_ca).idxmin()
obs_pos_ca, obs_vel_ca = np.split(df_observer.loc[t_closest].to_numpy(), 2)
obs_pos_ca = shift_pos(obs_pos_ca, obs_vel_ca, t_from=t_closest, t_to=t_ca)

ast_pos_init = np.array((-4.01739E+11, -6.48151E+10, 2.06501E+10))
ast_vel_init = np.array((4.60057E+03, -1.63450E+04, -3.81460E+02))
asteroid_name, df_asteroid = docks('asteroid', t0, t1, dt, ast_pos_init, ast_vel_init)
t_closest = abs(df_asteroid.index.to_series() - t_ca).idxmin()
ast_pos_ca, ast_vel_ca = np.split(df_asteroid.loc[t_closest].to_numpy(), 2)
ast_pos_ca = shift_pos(ast_pos_ca, ast_vel_ca, t_from=t_closest, t_to=t_ca)

sat_pos_ca, sat_vel_ca = params_to_coords(b_sat, v, alpha, beta, ast_pos_ca, ast_vel_ca, obs_pos_ca)
sat_pos_init = shift_pos(sat_pos_ca, sat_vel_ca, t_from=t_ca, t_to=t0)
_, df_spacecraft = docks('spacecraft', t0, t1, dt, sat_pos_init, sat_vel_ca, asteroid_name, round(GM, 1))
_, df_spacecraft_ref = docks('spacecraft_ref', t0, t1, dt, sat_pos_init, sat_vel_ca)

cub_pos_ca = (sat_pos_ca - ast_pos_ca) * (b_cub / b_sat) + ast_pos_ca
cub_pos_init = shift_pos(cub_pos_ca, sat_vel_ca, t_from=t_ca, t_to=t0)
_, df_cubesat = docks('cubesat', t0, t1, dt, cub_pos_init, sat_vel_ca, asteroid_name, round(GM, 1))
_, df_cubesat_ref = docks('cubesat_ref', t0, t1, dt, cub_pos_init, sat_vel_ca)

# Center on cubesat CA
t_closest = (df_cubesat - df_asteroid)[['x', 'y', 'z']].apply(np.linalg.norm, axis=1).idxmin()
obs_pos_ca, obs_vel_ca = np.split(df_observer.loc[t_closest].to_numpy(), 2)
ast_pos_ca, ast_vel_ca = np.split(df_asteroid.loc[t_closest].to_numpy(), 2)
cub_pos_ca, cub_vel_ca = np.split(df_cubesat.loc[t_closest].to_numpy(), 2)
cub_pos_ca, ast_pos_ca, delta_t = close_approach_calculator(cub_pos_ca, cub_vel_ca, ast_pos_ca, ast_vel_ca)
obs_pos_ca = shift_pos(obs_pos_ca, obs_vel_ca, delta_t=delta_t)
t_ca = t_closest + pd.to_timedelta(delta_t, 's')

b_, v_, alpha_, beta_ = coords_to_params(cub_pos_ca, cub_vel_ca, ast_pos_ca, ast_vel_ca, obs_pos_ca)
print(t_ca)
print(b_ / 1e3, v_ / 1e3)
print(alpha_ * 180. / pi, beta_ * 180. / pi)

df_observer = center_time_index(df_observer, t_ca)
df_asteroid = center_time_index(df_asteroid, t_ca)
df_spacecraft = center_time_index(df_spacecraft, t_ca)
df_satellite_ref = center_time_index(df_spacecraft_ref, t_ca)
df_cubesat = center_time_index(df_cubesat, t_ca)
df_cubesat_ref = center_time_index(df_cubesat_ref, t_ca)


# plot trajectory
fig1 = go.Figure(layout={'scene': {'aspectmode': 'data'}})
plot_trajectory(df_asteroid, 'Asteroid', fig1)
plot_trajectory(df_spacecraft, 'Spacecraft', fig1)
plot_trajectory(df_spacecraft_ref, 'Spacecraft (no asteroid)', fig1)
plot_trajectory(df_cubesat, 'cubesat', fig1)
plot_trajectory(df_cubesat_ref, 'cubesat (no asteroid)', fig1)
fig1.show()


# measurements
# ranging
df_relative = df_cubesat - df_spacecraft
rho = df_relative[['x', 'y', 'z']].apply(np.linalg.norm, axis=1)
tof = 2. * rho / c
# doppler frequency
v_r = (df_relative.vx * df_relative.x + df_relative.vy * df_relative.y + df_relative.vz * df_relative.z) / rho
freq = f0 * (1. - 2. * v_r / c)

# # measurement integration
# win_size = int(int_time / dt)
# freq = freq.rolling(win_size + (win_size % 2 == 0), center=True).mean().dropna().iloc[::win_size]
# tof = tof.rolling(win_size + (win_size % 2 == 0), center=True).mean().dropna().iloc[::win_size]
# # measurement noise
# freq += np.random.normal(scale=freq_noise, size=len(freq))
# tof += np.random.normal(scale=tof_noise, size=len(tof))

# freq_uncertainty = freq_noise
# tof_uncertainty = tof_noise

# Data gap
# t_full = freq.index.to_series()
# freq = freq.loc[t_full.abs() > 1. * 3600.]
# tof = tof.loc[t_full.abs() > 1. * 3600.]

# plot measurements
fig2 = go.Figure()
fig2.add_scatter(x=freq.index, y=freq, mode='markers', marker={'symbol': 'cross'}, name='doppler')
# fig2.show()
fig3 = go.Figure()
fig3.add_scatter(x=tof.index, y=tof, mode='markers', marker={'symbol': 'cross'}, name='time of flight')
# fig3.show()

# # # # # # # # # # # # # # # #
# Inversion

# Residuals

# residuals without asteroid
# ranging
df_relative = df_cubesat_ref - df_satellite_ref
rho = df_relative[['x', 'y', 'z']].apply(np.linalg.norm, axis=1)
tof_ref = 2. * rho / c
tof_residuals = tof - tof_ref
# doppler frequency
v_r = (df_relative.vx * df_relative.x + df_relative.vy * df_relative.y + df_relative.vz * df_relative.z) / rho
freq_ref = f0 * (1. - 2. * v_r / c)
freq_residuals = freq - freq_ref

# plot model
fig2.add_scatter(x=freq_ref.index, y=freq_ref, mode='lines', line={'color': 'red'}, name='doppler reference')
fig2.show()
fig3.add_scatter(x=tof_ref.index, y=tof_ref, mode='lines', line={'color': 'red'}, name='time of flight reference')
fig3.show()

# plot residuals
fig4 = go.Figure()
fig4.add_scatter(x=freq_residuals.index, y=freq_residuals, mode='markers',
                 marker={'symbol': 'cross', 'color': 'red'}, name='residuals')
fig4.show()

fig5 = go.Figure()
fig5.add_scatter(x=tof_residuals.index, y=tof_residuals, mode='markers',
                 marker={'symbol': 'cross', 'color': 'red'}, name='residuals')
fig5.show()
