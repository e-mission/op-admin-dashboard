# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - #
# Template Dashboard
# Basic dashboard to get started with
#
# Written by Kristi Potter
# June 29,2022
# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - #

# Dash/Plotly imports
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from plotly import graph_objs as go
import plotly.express as px
from dash.exceptions import PreventUpdate

# NREL branded component
import nrel_dash_components as ndc

# Data/file handling imports
import pathlib

# Etc
import pandas as pd

# Global data modules to share data across callbacks
# (Not sure how this stands up in multi-user/hosted situations)
import globals as gl
import globalsUpdater as gu

import os

#------------------------------------------------#
# The app and where you put any theme/style sheets
#------------------------------------------------#
app = dash.Dash(__name__,
                # You should use the dbc if you are using dbc components!
                external_stylesheets=[dbc.themes.BOOTSTRAP]
                )
server = app.server  # expose server variable for Procfile

#------------------------------------------------#
# Set the data path
#------------------------------------------------#

# For data that lives within the application.
# Set the path to the data directory
DATA_PATH = pathlib.Path(__file__).parent.joinpath("./data/").resolve()

#------------------------------------------------#
# DASH LAYOUT ROOT
#------------------------------------------------#
app.layout = ndc.NRELApp( # from: https://github.nrel.gov/nwunder2/nrel-dash-components
    appName="Template Dashboard Title",
    description="The title and this description are part of the nrel_dash_components.",

    # Add all children into this array
    children=[

        # A data store. This can save JSON-formats or simple data like booleans or integers
        dcc.Store(id='dataIsLoadedFlag', storage_type='memory'), # We can store JSON-izable data here, nothing too big

        # A section that stores some buttons
        html.Div(className="section notification", children=[

            "The crazy colors for each of the following elements are set by adding 'notification is-x' to the classname.",
            html.Br(),
            "Remove this classname modifier to get standard white background. ",
            html.Br(),
            "Having these colors is helpful when doing an initial layout.",
            html.Br(),html.Br(),

            # The P element can help us change the font (https://bulma.io/documentation/helpers/typography-helpers/)
            html.P(className="is-size-5 is-family-code" ,children=["Here's a set of tiles for buttons to load the data."]),
            # Tiles (from Bulma https://bulma.io/documentation/layout/tiles/) are easy ways to control layout.

            # Start with an ancestor tile
            html.Div(className="tile is-ancestor", children=[
                # Add a parent tile (children by default are laid out horizontally, for vertical layout, set is-vertical)
                html.Div(className="tile is-parent", children=[
                    # Then add children, set the size using bulma sizing (https://bulma.io/documentation/columns/sizes/), and color (https://bulma.io/documentation/helpers/color-helpers/)
                    html.Div(className="tile is-child is-half notification is-danger", children=[
                        "This is a tile with just this text and a button in it.",
                        html.Br(),
                        # Here's a button
                        html.Button("Load data", id="load-button",), # html button])
                    ]),
                    html.Div(className="tile is-child is-half notification is-info", children=[
                        "This tile has a vertical parent that lets us add buttons on top of each other.",

                        # Add a parent tile (children by default are laid out horizontally, for vertical layout, set is-vertical)
                        html.Div(className="tile is-parent is-vertical", children=[
                            # Create the buttons to load the map and graph
                            html.Div(className="tile is-child", id="load-map", children=[html.Button("Add data to map", id="map-button",style = dict(display='none'))]),
                            html.Div(className="tile is-child", id="load-graph", children=[html.Button("Add data to line chart", id="chart-button",style=dict(display='none'))]),
                        ])
                    ])
                ])
            ]),
        ]),
        # Make columns for the map and the line chart (look at the bulma.io documentation for reference)
        html.Div(className="columns", children=[
            html.Div(className="column is-6" , children=[
                # Map
                dcc.Graph(id='map',
                          config={'displayModeBar': False}, # Turns off the plotly figure interaction toolbar
                          )
            ] ),
            html.Div(className="column is-6 is-vcentered" , children=[

                # line graph
                dcc.Graph(id="line-graph") # This chart has the plotly toolbar
            ] ),
        ]),
        # Create a modal for errors and messages
        dbc.Modal([ dbc.ModalHeader("Modal Header", id="modal-header"),
                    dbc.ModalBody("This is the content of the modal", id="modal-body"),
                    dbc.ModalFooter(html.Button("Close", id="modal-close", n_clicks=0))],id="modal",is_open=False),
])


#------------------------------------------------#
# DASH CALLBACKS
#------------------------------------------------#
## --- Read in the data file, populate the charts --- ##
@app.callback(Output('dataIsLoadedFlag', 'data'),
              Output('map-button', 'style'),
              Output('chart-button', 'style'),
              Input('load-button', 'n_clicks'),
              prevent_initial_call=True  # All callbacks get called when the app is first loaded unless this is set
              )
def load_data(load_click):

    # Read in the data
    data = pd.read_csv(DATA_PATH.joinpath("rev_outs.csv"))

    # Set the global data
    gu.setDataStore(data)

    # Show the buttons
    style=dict(display='inline')

    # Signal that the data has been read
    return True, style, style

## --- Populate the map and the chart --- ##
@app.callback(Output('map', 'figure'),
              Output('line-graph', 'figure'),
              Input('map-button', 'n_clicks'),
              Input('chart-button', 'n_clicks'),
              State('dataIsLoadedFlag', 'data'),
              )
def update_charts(load, chart, haveDataState):

    # Change outputs based on which input is triggered
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'].split(".")[0]
    print("callback context: ", ctx.triggered)

    # Create a default map  (only return on intialize)
    map = px.scatter_mapbox(px.data.carshare(), lat="centroid_lat", lon="centroid_lon", # Datasframe, lat/lon column names
                            opacity=0, # This makes the data not visible
                            center={"lon": -96,"lat": 38}, zoom=2.75, # initial view state
                            mapbox_style ='carto-positron') # Free raster-tile
    # Update the layout of the map
    map.update_layout(margin=dict(l=0, r=0, t=0, b=0)) # Makes the map take up the whole div

    # The return value for a line chart
    fig = dash.no_update # As a return value, don't update the output

    # If we have loaded data, grab it from the global data variable
    data = None
    if(haveDataState):
        data = gl.dataStore

    # If we pushed the load data button create a map
    if(triggered == 'map-button'):

        # If we have no data yet, don't update anything
        # You could also put this as an else statement above,
        # but I'm putting it here so I can return an empty map
        # on the initial call, rather than an empty figure (like the line chart)
        if(data.empty):
            raise PreventUpdate()

        # Add the data to the map
        map = px.scatter_mapbox(data, lat="latitude", lon="longitude",
                                color="mean_cf",color_continuous_scale=px.colors.cyclical.IceFire,
                                center={"lon": -96,"lat": 38}, zoom=2.5,
                                mapbox_style ='carto-positron')
        # Update the layout of the map
        map.update_layout(margin=dict(l=0, r=0, t=0, b=0), # Makes the map take up the whole div
                          uirevision = 1) # Keeps the camera position when the map is updated

    # Else, return a line chart (only if data is previously loaded)
    elif(triggered =="chart-button"):

        # If we have no data yet, don't update anything
        # You could also put this as an else statement above,
        # but I'm putting it here so I can return an empty map
        # on the initial call, rather than an empty figure (like the line chart)
        if(data.empty):
            raise PreventUpdate()

        # Else, create the line figure, don't update the map
        fig = px.line(data,x='sc_gid',y='total_lcoe')
        map = dash.no_update


    # Return the map and the line figure
    return map, fig


## --- Populate the map and the chart --- ##
@app.callback(Output('modal', 'is_open'),
              Output('modal-header', 'children'),
              Output('modal-body', 'children'),
              Input('load-button', 'n_clicks'),
              Input('map-button', 'n_clicks'),
              Input('chart-button', 'n_clicks'),
              Input('modal-close', 'n_clicks'),
              State('dataIsLoadedFlag', 'data'),
              prevent_initial_call=True)
def updateModal(load, map, chart, close, dataState):

    # Change outputs based on which input is triggered
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'].split(".")[0]

    # Modal outputs
    modalOpen = dash.no_update
    modalHeader = dash.no_update
    modalBody = dash.no_update

    if(triggered == 'load-button') and (dataState != None):
        modalOpen = True
        modalHeader = "Oops!"
        modalBody = "Data already loaded."
    elif(triggered == 'map-button') and (map > 1):
        modalOpen = True
        modalHeader = "Oops!"
        modalBody = "Map data already loaded."
    elif(triggered =='chart-button') and (chart > 1):
        modalOpen = True
        modalHeader = "Oops!"
        modalBody = "Chart data already loaded."
    elif(triggered == 'modal-close'):
        modalOpen = False

    return  modalOpen, modalHeader, modalBody

#------------------------------------------------#
# PYTHON APP.PY Runs on Port 8050
#------------------------------------------------#
if __name__ == '__main__':
    envPort = int(os.getenv('SERVER_PORT', '8050'))
    envDebug = os.getenv('DASH_DEBUG_MODE', 'True').lower() == 'true'
    app.run(debug=envDebug, host='0.0.0.0', port=envPort)  # Set debug to False to remove that big blue dot
