from dash import dcc, html, Input, Output, callback, register_page, Dash
import pandas as pd
import pygwalker as pyg

from dash.dash import no_update
from utils import permissions as perm_utils


register_page(__name__, path="/charts")


# Define the layout
layout = html.Div([
    html.H1('PygWalker Interactive Charts', style={'textAlign': 'center'}),
    html.Div(
        "Select a dataset and visualize the data using PygWalker.",
        style={'textAlign': 'center', 'paddingBottom': '20px'}
    ),
    dcc.Dropdown(
        id='data-options',
        options=[
            {'label': 'Store Trips', 'value': 'store_trips'},
            {'label': 'Store User Stats', 'value': 'store_user_stats'}
        ],
        value='store_trips',
        style={'width': '50%', 'margin': '0 auto', 'paddingBottom': '20px'}
    ),
    dcc.Loading(
        id='loading-chart',
        type='circle',
        children=html.Iframe(
            id='charts',
            srcDoc='',
            style={'width': '100%', 'height': '80vh', 'border': 'none'}
        )
    )
], style={'padding': '20px'})

# Helper function to get data and permissions
def get_data_and_permissions(selected_option, store_user_stats, store_trips):
    if selected_option == 'store_trips':
        data = store_trips.get("data", None)
        columns = perm_utils.get_allowed_trip_columns().union(
            {col['label'] for col in perm_utils.get_allowed_named_trip_columns()}
        )
        has_perm = perm_utils.has_permission('data_trips')

    elif selected_option == 'store_user_stats':
        data = store_user_stats.get("data", None)
        columns = perm_utils.get_uuids_columns()
        has_perm = perm_utils.has_permission('data_uuids')

    else:
        data = None
        columns = set()
        has_perm = False

    return data, columns, has_perm

# Callback function to generate charts
@callback(
    Output('charts', 'srcDoc'),
    Input('store-user-stats', 'data'),
    Input('store-trips', 'data'),
    Input('data-options', 'value')
)
def generate_charts(store_user_stats, store_trips, selected_option):
    # Get data, columns, and permission status
    data, columns, has_perm = get_data_and_permissions(selected_option, store_user_stats, store_trips)

    if not data or not has_perm:
        return no_update

    # Create the DataFrame and filter columns based on permissions
    df = pd.DataFrame(data)
    allowed_columns = [col for col in df.columns if col in columns]
    df = df[allowed_columns]
    
    # Generate the PygWalker visualization
    walker = pyg.walk(
        df, 
        return_html=True, 
        hide_Data_Source_Config=True, 
        appearance='light'
    )
    
    # Ensure that walker is a string containing HTML
    if isinstance(walker, str):
        #print(walker)
        return walker
    elif hasattr(walker, 'to_html'):
        print(walker.to_html())
        return walker.to_html()
    else:
        # If walker is not a string or doesn't have to_html, return no_update
        return no_update
