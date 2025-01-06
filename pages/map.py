"""
Note that the callback will trigger even if prevent_initial_call=True. This is
because dcc.Location must be in app.py. Since the dcc.Location component is not
in the layout when navigating to this page, it triggers the callback. The
workaround is to check if the input value is None.
"""
from uuid import UUID

from dash import dcc, html, Input, Output, ALL, callback, register_page
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import logging

import emission.core.wrapper.user as ecwu
import emission.core.get_database as edb
import emcommon.diary.base_modes as emcdb

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

def create_single_option(value, color, label=None, icon=None):
    if icon:
        square = DashIconify(icon=f"mdi:{icon}", style={'margin': 'auto', 'color': color})
    else:
        square = html.Div(style={'background-color': color,
                                 'width': 10,
                                 'height': 10,
                                 'margin': 'auto'})
    
    text = html.Span(label or value,
                     style={'font': '14px monospace',
                            'color': color if color and icon else 'unset',
                            'maxHeight': 60,
                            'overflow': 'hidden'})
    return {
        'label': html.Span([square, text], style={'display': 'flex', 'gap': 5}),
        'value': value,
    }

def create_users_dropdown_options(trips_group_by_user_id, uuids_perm, tokens_perm):
    options = list()
    user_ids = set()
    for user_id in trips_group_by_user_id:
        color = trips_group_by_user_id[user_id]['color']
        if tokens_perm:
            token = ecwu.User.fromUUID(UUID(user_id))._User__email
            label = f"{token}\n({user_id})" if uuids_perm else token
        else:
            label = user_id if uuids_perm else ''
        user_ids.add(user_id)
        options.append(create_single_option(user_id, color, label=label))
    return options

def create_modes_dropdown_options(trips_group_by_user_mode):
    options = list()
    for mode in trips_group_by_user_mode:
        options.append(
            create_single_option(mode,
                                 trips_group_by_user_mode[mode]['color'],
                                 icon=trips_group_by_user_mode[mode]['icon'])
        )
    return options

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
                xl=3, lg=4, sm=6,
            )
        ]),
        dbc.Row(id='map-filters-row'),
        dbc.Row(id="trip-map-row")
    ],
    style={'display': 'flex', 'flex-direction': 'column', 'gap': 8}
)

@callback(
    Output('map-filters-row', 'children'),
    Input('map-type-dropdown', 'value'),
    Input('store-trips-map', 'data'),
)
def create_filters_dropdowns(map_type, trips_data):
    filters = []

    modes_options = create_modes_dropdown_options(
        trips_data['users_data_by_user_mode'],
    )
    filters.append(('Modes', 'modes', modes_options))

    uuids_perm, tokens_perm = has_permission('options_uuids'), has_permission('options_tokens')
    if map_type == 'lines' and (uuids_perm or tokens_perm):
        users_options = create_users_dropdown_options(
            trips_data['users_data_by_user_id'],
            uuids_perm,
            tokens_perm,
        )
        filters.append(('Users', 'users', users_options))

    return [
        dbc.Col(
            [
                html.Label(label),
                dcc.Dropdown(
                    id={'type': 'map-filter-dropdown', 'id': id},
                    options=options,
                    multi=True,
                    disabled=len(options) == 0,
                    optionHeight=(60 if id == 'users' else 35),
                )
            ],
            lg=4, sm=6,
        )
        for label, id, options in filters
    ]


@callback(
    Output('trip-map-row', 'children'),
    Input('map-type-dropdown', 'value'),
    Input({'type': 'map-filter-dropdown', 'id': ALL}, 'value'),
    Input('store-trips-map', 'data'),
)
def update_output(map_type, filter_values, trips_data):
    if not trips_data.get('users_data_by_user_id') and not trips_data.get('users_data_by_user_mode'):
        return html.Div("No trip data available for the selected date", id="no-trip-text", style={'font-size': 20, 'margin-top': 20})
    fig = None
    selected_modes = filter_values[0] if len(filter_values) > 0 else None
    selected_uuids = filter_values[1] if len(filter_values) > 1 else None
    coordinates = get_map_coordinates(trips_data['users_data_by_user_mode'], selected_modes)
    if map_type == 'lines':
        if selected_modes:
            fig = create_lines_map(
                coordinates,
                trips_data['users_data_by_user_mode'],
                selected_modes or [],
            )
        else:
            fig = create_lines_map(
                coordinates,
                trips_data['users_data_by_user_id'],
                selected_uuids or [],
            )
    elif map_type == 'heatmap':
        fig = create_heatmap_fig(coordinates)
    elif map_type == 'bubble':
        fig = create_bubble_fig(coordinates)
    else:
        fig = go.Figure()
    return dcc.Graph(figure=fig)


@callback(
    Output({'type': 'map-filter-dropdown', 'id': 'users'}, 'disabled'),
    Input({'type': 'map-filter-dropdown', 'id': 'modes'}, 'value'),
)
def control_user_dropdowns(selected_modes):
    return bool(selected_modes)


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
    Input('store-label-options', 'data'),
)
def store_trips_map_data(trips_data, label_options):
    if not trips_data['data']:
        return {'users_data_by_user_id': {}, 'users_data_by_user_mode': {}}
    
    trips_group_by_user_id = get_trips_group_by_user_id(trips_data)
    users_data_by_user_id = process_trips_group(trips_group_by_user_id)
  
    trips_group_by_user_mode = get_trips_group_by_user_mode(trips_data)
    users_data_by_user_mode = process_trips_group(trips_group_by_user_mode)
    if label_options:
        rich_modes = [[mode, emcdb.get_rich_mode_for_value(mode, label_options)]
                      for mode in users_data_by_user_mode]
        colors = [[mode, rich_mode['color']]
                  for mode, rich_mode in rich_modes]
        deduped_colors = emcdb.dedupe_colors(colors, adjustment_range=[0.5,1.5])
        for i, (mode, trips_group) in enumerate(users_data_by_user_mode.items()):
            trips_group['icon'] = rich_modes[i][1]['icon']
            trips_group['color'] = deduped_colors[mode]

    return {'users_data_by_user_id':users_data_by_user_id, 'users_data_by_user_mode':users_data_by_user_mode}
