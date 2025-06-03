import numpy as np
from scipy.signal import butter, filtfilt
from dash import Dash, dcc, html, Output, Input, State
import plotly.graph_objs as go
import dash

# Константи
defaults = {
    'amp': 1.0,
    'freq': 1.0,
    'phi': 0.0,
    'noise_mean': 0.0,
    'noise_var': 0.1,
    'filter_order': 4,
    'cutoff': 5.0,
    'show_options': ['show_noise', 'show_filter']
}

t = np.linspace(0, 2, 2000)

def generate_y(amp, freq, phi):
    return amp * np.sin(2 * np.pi * freq * t + phi)

def generate_noise(mean, var):
    return np.random.normal(mean, np.sqrt(var), len(t))

def filter_signal(signal, order, cutoff):
    fs = 1 / (t[1] - t[0])
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='main-plot'),
    dcc.Store(id='noise-data'),

    html.Div([
        html.Label('Амплітуда'), dcc.Slider(0.1, 5.0, 0.01, value=defaults['amp'], id='amp', marks={0.1: '0.1', 5: '5'}),

        html.Label('Частота'), dcc.Slider(0.1, 5.0, 0.01, value=defaults['freq'], id='freq', marks={0.1: '0.1', 5: '5'}),

        html.Label('Фазовий зсув'), dcc.Slider(-2*np.pi, 2*np.pi, 0.01, value=defaults['phi'], id='phi', marks={-2*np.pi: '-2pi', 2*np.pi: '2pi'}),

        html.Label('Середнє шуму'), dcc.Slider(-1.0, 1.0, 0.01, value=defaults['noise_mean'], id='noise_mean', marks={-1: '-1', 1: '1'}),

        html.Label('Дисперсія шуму'), dcc.Slider(0.0, 1.0, 0.01, value=defaults['noise_var'], id='noise_var', marks={0: '0', 1: '1'}),

        html.Label('Порядок фільтра'), dcc.Slider(1, 9, 1, value=defaults['filter_order'], id='filter_order'),

        html.Label('Частота зрізу'), dcc.Slider(1, 9, 0.1, value=defaults['cutoff'], id='cutoff', marks={1: '1', 9: '9'}),

        html.Br(),
        html.Button('Скинути', id='reset', n_clicks=0),

        dcc.Checklist(
            options=[{'label': 'Шум', 'value': 'show_noise'}, {'label': 'Фільтр', 'value': 'show_filter'}],
            value=defaults['show_options'], id='show_options', inline=True
        ),
    ], style={'width': '50%', 'margin': 'auto'})
])

@app.callback(
    Output('main-plot', 'figure'),

    Output('amp', 'value'),
    Output('freq', 'value'),
    Output('phi', 'value'),

    Output('noise_mean', 'value'),
    Output('noise_var', 'value'),

    Output('filter_order', 'value'),
    Output('cutoff', 'value'),

    Output('show_options', 'value'),
    Output('noise-data', 'data'),

    Input('amp', 'value'),
    Input('freq', 'value'),
    Input('phi', 'value'),

    Input('noise_mean', 'value'),
    Input('noise_var', 'value'),

    Input('filter_order', 'value'),
    Input('cutoff', 'value'),

    Input('show_options', 'value'),
    Input('reset', 'n_clicks'),

    State('noise-data', 'data')
)
def update_plot(amp, freq, phi, noise_mean, noise_var, order, cutoff, show_options, reset_clicks, noise_data):
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id']
        triggered = prop_id.split('.')[0]
    else:
        triggered = None

    if triggered == 'reset':
        amp, freq, phi = defaults['amp'], defaults['freq'], defaults['phi']

        noise_mean, noise_var = defaults['noise_mean'], defaults['noise_var']

        order, cutoff = defaults['filter_order'], defaults['cutoff']

        show_options = defaults['show_options']

        noise = generate_noise(noise_mean, noise_var)
    else:
        if triggered == 'noise_mean' or triggered == 'noise_var' or noise_data is None:
            noise = generate_noise(noise_mean, noise_var)
        else:
            noise = np.array(noise_data)

    y = generate_y(amp, freq, phi)
    y_filtered = filter_signal(y + noise, order, cutoff)

    traces = []

    if 'show_noise' in show_options:
        traces.append(go.Scatter(x=t, y= y + noise, mode='lines', name='Сигнал + шум', opacity=0.3, showlegend=True))

    traces.append(go.Scatter(x=t, y=y, mode='lines', name='Сигнал'))

    if 'show_filter' in show_options:
        traces.append(go.Scatter(x=t, y=y_filtered, mode='lines', name='Фільтрований сигнал'))

    fig = go.Figure(traces)
    fig.update_layout(
        xaxis_title='t', yaxis_title='y(t)',
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(x=0, y=1)
    )

    return fig, amp, freq, phi, noise_mean, noise_var, order, cutoff, show_options, noise.tolist()

if __name__ == '__main__':
    app.run(debug=True)
