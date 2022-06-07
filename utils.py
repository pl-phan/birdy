import itertools

import numpy as np
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

color_iterator = itertools.cycle(DEFAULT_PLOTLY_COLORS)
c = 3e8  # m/s


def unit_vector(alpha, beta):
    return Rotation.from_euler('ZY', (alpha, beta)).apply(np.array((1., 0., 0.)))


def next_color():
    return next(color_iterator)


def show_covariance(mu, cov, names, units=None, true_values=None):
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
                    fig.add_vline(x=true_values[i], col=i + 1, row=j + 1, line={'color': 'green'})
                else:
                    fig.add_scatter(x=true_values[i:i+1], y=true_values[j:j+1], col=i + 1, row=j + 1,
                                    marker={'color': 'green', 'symbol': 'cross'})
    fig.update_layout(title='Result distribution', showlegend=False, coloraxis=dict(colorscale='greys'))
    fig.update_coloraxes(showscale=False)
    fig.show()


def plot_trajectories(trajectories):
    fig = go.Figure()
    for traj in trajectories:
        fig.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], mode='lines+markers'))
    # fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()


if __name__ == '__main__':
    # TODO TESTS
    show_covariance((100., 5.), ((3., -1.5), (-1.5, 1.)), names=('p1', 'p2'))
