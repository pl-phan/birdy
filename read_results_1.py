import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import sci_notation

ms_table = pd.read_csv('./v_b_m5p_backup.csv')
vs = ms_table.v.sort_values().unique()
bs = ms_table.b.sort_values().unique()

ms = np.empty((len(bs), len(vs)))
for j, v in enumerate(vs):
    for i, b in enumerate(bs):
        ms[i, j] = ms_table.loc[(ms_table.v == v) & (ms_table.b == b)].iloc[0]['m_5p']

fig = go.Figure()
ticks = np.arange(13, 17.5, step=0.5)
fig.add_trace(go.Contour(
    x=vs / 1000., y=bs / 1000., z=np.log10(ms), line={'width': 2},
    contours={'coloring': 'lines', 'showlabels': True, 'start': 13, 'size': 0.5, 'end': 17},
    colorbar={'tickvals': ticks, 'ticktext': [sci_notation(10 ** t) + 'kg' for t in ticks], 'orientation': 'h'},
    zmin=13., zmax=18., colorscale=[[0., 'black'], [1., 'black']],
))
ticks = np.arange(13, 19, step=1.)
fig.add_trace(go.Heatmap(
    x=vs / 1000., y=bs / 1000., z=np.log10(ms),
    colorbar={'tickvals': ticks, 'ticktext': [sci_notation(10 ** t) + 'kg' for t in ticks]},
    zmin=13., zmax=18., colorscale='Jet'
))

fig.update_xaxes(type='log')
fig.update_yaxes(type='log')
fig.update_layout(
    xaxis_title='Velocity (km/s)',
    yaxis_title='Distance (km)',
    font={'size': 20}
)
fig.show()
