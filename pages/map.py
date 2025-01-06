"""
Note that the callback will trigger even if prevent_initial_call=True. This is
because dcc.Location must be in app.py. Since the dcc.Location component is not
in the layout when navigating to this page, it triggers the callback. The
workaround is to check if the input value is None.
"""
from uuid import UUID

from dash import dcc, html, Input, Output, State, callback, register_page
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import logging

import emission.core.wrapper.user as ecwu
import emission.core.get_database as edb
import logging

from utils.permissions import has_permission

register_page(__name__, path="/map")

intro = """## Map"""


def create_lines_map(coordinates, trips_group_by_user_id, user_id_list):
    traces = []
    for user_id in user_id_list:
        color = trips_group_by_user_id[user_id]['color']
        trips = trips_group_by_user_id[user_id]['trips']

        for trip in trips:
            traces.append(
                go.Scattermapbox(
                    mode="markers+lines",
                    lon=[trip['start_coordinates'][0], trip['end_coordinates'][0]],
                    lat=[trip['start_coordinates'][1], trip['end_coordinates'][1]],
                    marker={'size': 10, 'color': color},
                )
            )

    fig = go.Figure(data=traces)
    (zoom, center) = get_mapbox_zoom_and_center(coordinates)
    fig.update_layout(
        showlegend=False,
        margin={'l': 0, 't': 30, 'b': 0, 'r': 0},
        mapbox_style="open-street-map",
        mapbox_center_lon=center[0],
        mapbox_center_lat=center[1],
        mapbox_zoom=zoom,
        height=650,
    )

    return fig


def get_map_coordinates(trips_group_by_user_mode, user_mode_list):
    coordinates = {'lat': [], 'lon': [], 'color':[]}
    for user_mode, group in trips_group_by_user_mode.items():
        if user_mode_list and user_mode not in user_mode_list:
            continue
        for trip in group['trips']:
            coordinates['lon'].append(trip['start_coordinates'][0])
            coordinates['lon'].append(trip['end_coordinates'][0])
            coordinates['lat'].append(trip['start_coordinates'][1])
            coordinates['lat'].append(trip['end_coordinates'][1])
            coordinates['color'].extend([group['color'], group['color']])
    return coordinates


def create_heatmap_fig(coordinates):
    fig = go.Figure()
    if len(coordinates.get('lat', [])) > 0:
        fig.add_trace(
            go.Densitymapbox(
                lon=coordinates['lon'],
                lat=coordinates['lat'],
                name = '',
                
            )
        )
        (zoom, center) = get_mapbox_zoom_and_center(coordinates)
        fig.update_layout(
            mapbox_style='open-street-map',
            mapbox_center_lon=center[0],
            mapbox_center_lat=center[1],
            mapbox_zoom=zoom,
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            height=650,
        )
    return fig


# derived from https://community.plotly.com/t/dynamic-zoom-for-mapbox/32658/12
# workaround until dash team implements dynamic zoom on mapbox Scattermapbox and Densitymapbox
def get_mapbox_zoom_and_center(coords):
    if (not coords.get('lon') or not coords.get('lat') or len(coords['lon']) != len(coords['lat'])):
        logging.error("Invalid input to get_mapbox_zoom_and_center, coords: " + str(coords))
        return 0, (0, 0)

    min_lonlat = (min(coords['lon']), min(coords['lat']))
    max_lonlat = (max(coords['lon']), max(coords['lat']))
    midpoint = (min_lonlat[0] + max_lonlat[0]) / 2, (min_lonlat[1] + max_lonlat[1]) / 2
    area = (max_lonlat[0] - min_lonlat[0]) * (max_lonlat[1] - min_lonlat[1])
    zoom = np.interp(x=area,
                      xp=[0, 5**-10, 4**-10, 3**-10, 2**-10, 1**-10, 1**-5],
                      fp=[20, 15,    14,     13,     12,     7,      5])
    zoom = int(min(15, zoom))
    logging.debug("zoom: " + str(zoom) + " midpoint: " + str(midpoint))
    return zoom, midpoint


def create_bubble_fig(coordinates):
    fig = go.Figure()
    if len(coordinates.get('lon', [])) > 0:
        fig.add_trace(
            go.Scattermapbox(
                lat=coordinates['lat'],
                lon=coordinates['lon'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=9,
                    color=coordinates['color'],
                ),
            )
        )
        (zoom, center) = get_mapbox_zoom_and_center(coordinates)
        fig.update_layout(
            autosize=True,
            mapbox_style='open-street-map',
            mapbox_center_lon=center[0],
            mapbox_center_lat=center[1],
            mapbox_zoom=zoom,
            mapbox_bearing=0,
            margin={'r': 0, 't': 30, 'l': 0, 'b': 0},
            height=650,
        )
    return fig


def get_trips_group_by_user_id(trips_data):
    trips_group_by_user_id = None
    trips_df = pd.DataFrame(trips_data['data'])
    if not trips_df.empty:
        trips_group_by_user_id = trips_df.groupby('user_id')
    return trips_group_by_user_id

def get_trips_group_by_user_mode(trips_data):
    trips_group_by_user_mode = None
    trips_df = pd.DataFrame(trips_data.get('data'))
    if not trips_df.empty:
        if 'mode_confirm' not in trips_df.columns:
            trips_df['mode_confirm'] = None
        trips_df['mode_confirm'] = trips_df['mode_confirm'].fillna('Unlabeled')
        trips_group_by_user_mode = trips_df.groupby('mode_confirm')
    return trips_group_by_user_mode

def create_single_option(value, color):
    return {
        'label': html.Span(
            [
                html.Div(id='dropdown-squares', style={'background-color': color}),
                html.Span(value, style={'font-size': 15, 'padding-left': 10})
            ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
        ),
        'value': value
    }

def create_user_ids_options(trips_group_by_user_id):
    options = list()
    user_ids = set()
    if has_permission('options_uuids'):
        for user_id in trips_group_by_user_id:
            color = trips_group_by_user_id[user_id]['color']
            user_ids.add(user_id)
            options.append(create_single_option(user_id, color))
    return options, user_ids

def create_user_emails_options(trips_group_by_user_id):
    options = list()
    user_emails = set()
    if has_permission('options_emails'):
        for i, user_id in enumerate(trips_group_by_user_id):
            color = trips_group_by_user_id[user_id]['color']
            logging.warn("dict is %s" % ecwu.User.fromUUID(UUID(user_id)).__dict__)
            logging.warn("all users are %s" % list(edb.get_uuid_db().find()))
            try:
                user_email = ecwu.User.fromUUID(UUID(user_id))._User__email
            except AttributeError as e:
                continue
            user_emails.add(user_email)
            options.append(create_single_option(user_email, color))
    return options, user_emails

def create_user_modes_options(trips_group_by_user_mode):
    options = list()
    user_modes = set()
    for user_mode in trips_group_by_user_mode:
        color = trips_group_by_user_mode[user_mode]['color']
        user_modes.add(user_mode)
        options.append(create_single_option(user_mode, color))
    return options, user_modes

map_type_options = []
if has_permission('map_heatmap'):
    map_type_options.append({'label': 'Density Heatmap', 'value': 'heatmap'})
if has_permission('map_bubble'):
    map_type_options.append({'label': 'Bubble Map', 'value': 'bubble'})
if has_permission('map_trip_lines'):
    map_type_options.append({'label': 'Trips Lines', 'value': 'lines'})


layout = html.Div(
    [
        dcc.Store(id="store-trips-map", data={}),
        dcc.Markdown(intro),

        dbc.Row([
            dbc.Col(
                [
                    html.Label('Map Type'),
                    dcc.Dropdown(
                        id='map-type-dropdown',
                        options=map_type_options,
                        value=map_type_options[0]['value'],
                    ),
                ],
                xl=3,
                lg=4,
                sm=6,
            )
        ]),

        dbc.Row([
            dbc.Col([
                html.Label('User UUIDs'),
                dcc.Dropdown(id='user-id-dropdown', multi=True),
            ], style={'display': 'block' if has_permission('options_uuids') else 'none'}),
            dbc.Col([
                html.Label('User Emails'),
                dcc.Dropdown(id='user-email-dropdown', multi=True),
            ], style={'display': 'block' if has_permission('options_emails') else 'none'}),
            dbc.Col([
                html.Label('Modes'),
                dcc.Dropdown(id='user-mode-dropdown', multi=True),
            ], style={'display': 'block'})
        ]),
        dbc.Row(id="trip-map-row")
    ]
)

@callback(
    Output('user-id-dropdown', 'options'),
    Output('user-id-dropdown', 'value'),
    Input('store-trips-map', 'data'),
    Input('user-id-dropdown', 'value'),
)
def update_user_ids_options(trips_data, selected_user_ids):
    user_ids_options, user_ids = create_user_ids_options(trips_data['users_data_by_user_id'])
    if selected_user_ids is not None:
        selected_user_ids = [user_id for user_id in selected_user_ids if user_id in user_ids]
    return user_ids_options, selected_user_ids


@callback(
    Output('user-email-dropdown', 'options'),
    Output('user-email-dropdown', 'value'),
    Input('store-trips-map', 'data'),
    Input('user-email-dropdown', 'value'),
)
def update_user_emails_options(trips_data, selected_user_emails):
    user_emails_options, user_emails = create_user_emails_options(trips_data['users_data_by_user_id'])
    if selected_user_emails is not None:
        selected_user_emails = [user_email for user_email in selected_user_emails if user_email in user_emails]
    return user_emails_options, selected_user_emails

@callback(
    Output('user-mode-dropdown', 'options'),
    Output('user-mode-dropdown', 'value'),
    Output('user-mode-dropdown', 'disabled'),
    Input('store-trips-map', 'data'),
    Input('user-mode-dropdown', 'value'),
)
def update_user_modes_options(trips_data, selected_user_modes):
    user_modes_options, user_modes = create_user_modes_options(trips_data['users_data_by_user_mode'])
    if selected_user_modes is not None:
        selected_user_modes = [mode_confirm for mode_confirm in selected_user_modes if mode_confirm in user_modes]
    
    # Disable the 'mode' button if no user_modes_options have 0 values
    return user_modes_options, selected_user_modes, len(user_modes_options) == 0

@callback(
    Output('trip-map-row', 'children'),
    Input('map-type-dropdown', 'value'),
    Input('user-id-dropdown', 'value'),
    Input('user-email-dropdown', 'value'),
    Input('user-mode-dropdown', 'value'),
    State('store-trips-map', 'data'),
)
def update_output(map_type, selected_user_ids, selected_user_emails, selected_user_modes, trips_data):
    if not trips_data['users_data_by_user_id'] and not trips_data['users_data_by_user_mode']:
        return html.Div("No trip data available for the selected date", id="no-trip-text", style={'font-size': 20, 'margin-top': 20})
    fig = None
    user_ids = set(selected_user_ids) if selected_user_ids is not None else set()
    user_modes=set(selected_user_modes) if selected_user_modes is not None else set()
    coordinates = get_map_coordinates(trips_data.get('users_data_by_user_mode', {}), user_modes)
    if selected_user_emails is not None:
        for user_email in selected_user_emails:
            user_ids.add(str(ecwu.User.fromEmail(user_email).uuid))
    if map_type == 'lines':
        if selected_user_modes:
            fig = create_lines_map(trips_data.get('users_data_by_user_mode', {}), user_modes)
        else:
            fig = create_lines_map(trips_data.get('users_data_by_user_id', {}), user_ids)
    elif map_type == 'heatmap':
        fig = create_heatmap_fig(coordinates)
    elif map_type == 'bubble':
        fig = create_bubble_fig(coordinates)
    else:
        fig = go.Figure()
    return dcc.Graph(figure=fig)


@callback(
    Output('user-id-dropdown', 'disabled'),
    Output('user-email-dropdown', 'disabled'),
    Input('map-type-dropdown', 'value'),
    Input('user-mode-dropdown', 'value'),
)
def control_user_dropdowns(map_type,selected_user_modes):
    disabled = True
    if map_type == 'lines':
        disabled = False
        if selected_user_modes:
            disabled = True
    return disabled, disabled


def process_trips_group(trips_group):
    users_data = dict()
    #processes a group of trips, assigns color to each group and stores the processed data in a dictionary
    if trips_group:
        keys = list(trips_group)
        n = len(keys) % 360
        k = 359 // (n - 1) if n > 1 else 0
        for ind, key in enumerate(trips_group.groups.keys()):
            color = f'hsl({ind * k}, 100%, 50%)'
            trips = trips_group.get_group(key).to_dict("records")
            users_data[key] = {'color': color, 'trips': trips}  
    return users_data


@callback(
    Output('store-trips-map', 'data'),
    Input('store-trips', 'data'),
)
def store_trips_map_data(trips_data):
    if not trips_data['data']:
        return {'users_data_by_user_id': {}, 'users_data_by_user_mode': {}}
    
    trips_group_by_user_id = get_trips_group_by_user_id(trips_data)
    users_data_by_user_id = process_trips_group(trips_group_by_user_id)
  
    trips_group_by_user_mode = get_trips_group_by_user_mode(trips_data)
    users_data_by_user_mode = process_trips_group(trips_group_by_user_mode)

    return {'users_data_by_user_id':users_data_by_user_id, 'users_data_by_user_mode':users_data_by_user_mode}
