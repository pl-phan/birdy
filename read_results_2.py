import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import sci_notation


ms_table = pd.read_csv('./bx_by_m5p_10000_3000000_n98_backup.csv')
bxs = ms_table.bx.sort_values().unique()
bys = ms_table.by.sort_values().unique()
bys = np.concatenate((bys, -bys[::-1]))

ms = np.empty((len(bys), len(bxs)))
for j, bx in enumerate(bxs):
    for i, by in enumerate(bys):
        ms[i, j] = ms_table.loc[(ms_table.bx == bx) & (ms_table.by == -abs(by))].iloc[0]['m_5p']

fig = go.Figure()
ticks = np.arange(15, 17.2, step=0.2)
fig.add_trace(go.Contour(
    x=bxs / 1000., y=bys / 1000., z=np.log10(ms), line={'width': 2},
    contours={'coloring': 'lines', 'showlabels': True, 'start': 15, 'size': 0.2, 'end': 17},
    colorbar={'tickvals': ticks, 'ticktext': [sci_notation(10 ** t) + 'kg' for t in ticks], 'orientation': 'h'},
    zmin=15., zmax=18., colorscale=[[0., 'black'], [1., 'black']],
))
ticks = np.arange(15, 18.2, step=0.5)
fig.add_trace(go.Heatmap(
    x=bxs / 1000., y=bys / 1000., z=np.log10(ms),
    colorbar={'tickvals': ticks, 'ticktext': [sci_notation(10 ** t) + 'kg' for t in ticks]},
    zmin=15., zmax=18., colorscale='Jet'
))

fig.update_yaxes(scaleanchor='x', scaleratio=1)
fig.update_layout(
    xaxis_title='B-plane x-coordinate (km)',
    yaxis_title='B-plane y-coordinate (km)',
    font={'size': 18}
)
fig.show()
