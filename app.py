import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import us

mapbox_access_token = "pk.eyJ1IjoicHJpeWF0aGFyc2FuIiwiYSI6ImNqbGRyMGQ5YTBhcmkzcXF6YWZldnVvZXoifQ.sN7gyyHTIq1BSfHQRBZdHA"

df1 = pd.read_csv('C://Ashish/Project/dataset/visualization-ready/r2v-data-irma2-states.csv', nrows = 10)
df = df1.dropna(axis=0)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Hurricane Irma - United States")
    ], style={
        'textAlign': "center",
        "padding-bottom": "10",
        "padding-top": "10"}),
    html.Div([
        dcc.Dropdown(id="state-selected",
                     options=[{'label': f'{us.states.lookup(i)}', 'value': i} for i in df.state.unique()],
                     value=['CA'],
                     multi=True,
                     style={
                         "display": "block",
                         "margin-left": "auto",
                         "margin-right": "auto",
                         "width": "50%"

                     }
                     )
    ]),
    html.Div(dcc.Graph(id="my-graph"))

])


@app.callback(
    dash.dependencies.Output("my-graph", "figure"),
    [dash.dependencies.Input("state-selected", "value")]

)
def update_figure(selected):
    trace = []
    for state in selected:
        dff = df[df["state"] == state]
        trace.append(go.Scattermapbox(
            lat=dff["lat"],
            lon=dff["long"],
            mode='markers',
            marker={'symbol': "square", 'size': 10},
            text=dff['original_text'],
            hoverinfo='text',
            name=state
        ))
    return {
        "data": trace,
        "layout": go.Layout(
            autosize=True,
            hovermode='closest',
            showlegend=False,
            height=700,
            mapbox={'accesstoken': mapbox_access_token,
                    'bearing': 0,
                    'center': {'lat': 38, 'lon': -94},
                    'pitch': 30, 'zoom': 3,
                    "style": 'mapbox://styles/mapbox/light-v9'},
        )

    }


server = app.server

if __name__ == '__main__':
	app.run_server(debug=False)