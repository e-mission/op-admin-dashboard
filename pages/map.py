"""
Note that the callback will trigger even if prevent_initial_call=True. This is
because dcc.Location must be in app.py. Since the dcc.Location component is not
in the layout when navigating to this page, it triggers the callback. The
workaround is to check if the input value is None.
"""
import logging
import random
from uuid import UUID
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import dcc, html, Input, Output, ALL, callback, register_page, callback_context
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

import emission.core.wrapper.user as ecwu
import emission.core.get_database as edb
import emcommon.diary.base_modes as emcdb
import emcommon.bluetooth.ble_matching as emcble
import emission.analysis.configs.dynamic_config as eacd

from utils.permissions import has_permission

config = eacd.get_dynamic_config()

################################################################################
# Create a lookup from baseMode => list of vehicle IDs
################################################################################
def create_baseMode_lookup():
    """
    Returns a dict like:
        {
          "CAR":   ["car_jacks_mazda3", "car_abbys_prius", ...],
          "E_CAR": ["ecar_gsa_leaf1", "ecar_leidy_car", ...],
          "WHEELCHAIR": [...],
          ...
        }
    """
    vehicle_identities = config.get("vehicle_identities", [])
    base_mode_dict = defaultdict(list)
    for veh in vehicle_identities:
        base_mode = veh.get("baseMode")   # e.g. "CAR"
        veh_id    = veh.get("value")      # e.g. "car_jacks_mazda3"
        if base_mode and veh_id:
            base_mode_dict[base_mode].append(veh_id)
    return dict(base_mode_dict)

baseMode_to_vehicles = create_baseMode_lookup()

def get_base_mode_for_trip(trip):
    """
    Return the base mode (e.g. 'CAR', 'E_CAR', 'WHEELCHAIR') for this trip,
    if it can be inferred from trip["data.primary_ble_sensed_mode"].
    Otherwise return None if unknown.
    """
    raw_mode = trip.get("data.primary_ble_sensed_mode")
    if not raw_mode or raw_mode == "UNKNOWN":
        return None

    # If raw_mode is already a known base mode, return it
    if raw_mode in baseMode_to_vehicles:
        return raw_mode

    # Otherwise, raw_mode should be a specific vehicle ID like "car_jacks_mazda3".
    # Let's see which base mode it belongs to.
    for base_mode, vehicle_ids in baseMode_to_vehicles.items():
        if raw_mode in vehicle_ids:
            return base_mode

    # If we can't find a matching base mode, treat it as None
    return None


################################################################################
# Helper to filter trips by user-chosen BLE base modes (CAR, E_CAR, etc.)
################################################################################
def filter_by_ble_modes(data_dict, ble_base_modes_selected):
    """
    For each trip, figure out the trip's base mode and check if that base mode
    is in ble_base_modes_selected. Keep the trip if yes.
    """
    if not ble_base_modes_selected:
        logging.info("No BLE modes selected => returning data unfiltered")
        return {}
    filtered = {}
    for group_key, group_val in data_dict.items():
        orig_trips = group_val['trips']
        logging.info(f"Filtering group '{group_key}' with {len(orig_trips)} total trips (BLE)")
        filtered_trips = []
        for trip in orig_trips:
            base_mode = get_base_mode_for_trip(trip)
            # Keep the trip if the trip's base_mode is in the user selection
            if base_mode and base_mode in ble_base_modes_selected:
                filtered_trips.append(trip)
        logging.info(f"After BLE filtering, group '{group_key}' => {len(filtered_trips)} trips remain")
        if filtered_trips:
            # Shallow copy the group's metadata
            filtered[group_key] = dict(group_val)
            filtered[group_key]['trips'] = filtered_trips

    return filtered

################################################################################
# Helper to filter trips by sensed modes
################################################################################
def filter_by_sensed_modes(data_dict, sensed_modes_selected):
    if not sensed_modes_selected:
        logging.info("No sensed modes selected => returning data unfiltered")
        return {}
    filtered = {}
    for group_key, group_val in data_dict.items():
        orig_trips = group_val['trips']
        logging.info(f"Filtering group '{group_key}' with {len(orig_trips)} total trips (Sensed)")
        filtered_trips = []
        for trip in orig_trips:
            sensed_mode = trip.get("data.primary_sensed_mode")
            if sensed_mode and sensed_mode in sensed_modes_selected:
                filtered_trips.append(trip)
        logging.info(f"After sensed filtering, group '{group_key}' => {len(filtered_trips)} trips remain")
        if filtered_trips:
            filtered[group_key] = dict(group_val)
            filtered[group_key]['trips'] = filtered_trips
    return filtered

################################################################################
# Begin page definitions
################################################################################
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
    """
    Build arrays of lat, lon, color, and text so that the bubble map can
    display detailed hover info (including base BLE mode) for each start/end.
    """
    coordinates = {
        'lat': [],
        'lon': [],
        'color': [],
        'text': [],  # We'll store the hover text for each point here
    }

    for user_mode, group in trips_group_by_user_mode.items():
        if user_mode_list and user_mode not in user_mode_list:
            continue

        for trip in group['trips']:
            # Basic info from the trip dict
            user_id = trip.get('user_id', 'Unknown user')
            start_time = trip.get('trip_start_time_str', 'No start time')
            end_time = trip.get('trip_end_time_str', 'No end time')
            distance_m = trip.get('data.distance_meters', 0)
            mode_label = trip.get('mode_confirm', 'Unlabeled')

            # NEW: get the base BLE mode (CAR, E_CAR, etc.)
            base_mode = get_base_mode_for_trip(trip)
            ble_str = base_mode if base_mode else "None"
            sensed_mode = trip.get("data.primary_sensed_mode", "None")
            start_hover_text = (
                f"<b>User:</b> {user_id}<br>"
                f"<b>Mode:</b> {mode_label}<br>"
                f"<b>BLE Mode:</b> {ble_str}<br>"
                f"<b>Sensed Mode:</b> {sensed_mode}<br>"
                f"<b>Started:</b> {start_time}<br>"
                f"<b>Distance (m):</b> {round(distance_m, 2)}"
            )
            end_hover_text = (
                f"<b>User:</b> {user_id}<br>"
                f"<b>Mode:</b> {mode_label}<br>"
                f"<b>BLE Mode:</b> {ble_str}<br>"
                f"<b>Sensed Mode:</b> {sensed_mode}<br>"
                f"<b>Ended:</b> {end_time}<br>"
                f"<b>Distance (m):</b> {round(distance_m, 2)}"
            )
            coordinates['lon'].extend([trip['start_coordinates'][0], trip['end_coordinates'][0]])
            coordinates['lat'].extend([trip['start_coordinates'][1], trip['end_coordinates'][1]])
            coordinates['color'].extend([group['color'], group['color']])
            coordinates['text'].extend([start_hover_text, end_hover_text])
    return coordinates

def create_heatmap_fig(coordinates):
    fig = go.Figure()
    if coordinates.get('lat'):
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
    if coordinates.get('lon'):
        fig.add_trace(
            go.Scattermapbox(
                lat=coordinates['lat'],
                lon=coordinates['lon'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=9,
                    color=coordinates['color'],
                ),
                text=coordinates['text'],
                hoverinfo='text'
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
    trips_df = pd.DataFrame(trips_data['data'])
    return trips_df.groupby('user_id') if not trips_df.empty else None

def get_trips_group_by_user_mode(trips_data):
    trips_df = pd.DataFrame(trips_data.get('data'))
    if not trips_df.empty:
        if 'mode_confirm' not in trips_df.columns:
            trips_df['mode_confirm'] = None
        trips_df['mode_confirm'] = trips_df['mode_confirm'].fillna('Unlabeled')
        return trips_df.groupby('mode_confirm')
    return None

def create_single_option(value, color, label=None, icon=None):
    if icon:
        square = DashIconify(icon=f"mdi:{icon}", style={'margin': 'auto', 'color': color})
    else:
        square = html.Div(
            style={
                'background-color': color,
                'width': 10,
                'height': 10,
                'margin': 'auto'
            }
        )

    text = html.Span(
        label or value,
        style={
            'font': '14px monospace',
            'color': color if color and icon else 'unset',
            'maxHeight': 60,
            'overflow': 'hidden'
        }
    )
    return {
        'label': html.Span([square, text], style={'display': 'flex', 'gap': 5}),
        'value': value,
    }

def create_users_dropdown_options(trips_group_by_user_id, uuids_perm, tokens_perm):
    options = []
    for user_id in trips_group_by_user_id:
        color = trips_group_by_user_id[user_id]['color']
        if tokens_perm:
            token = ecwu.User.fromUUID(UUID(user_id))._User__email
            label = f"{token}\n({user_id})" if uuids_perm else token
        else:
            label = user_id if uuids_perm else ''
        options.append(create_single_option(user_id, color, label=label))
    return options

def create_modes_dropdown_options(trips_group_by_user_mode):
    options = list()
    for mode in trips_group_by_user_mode:
        logging.info(f'trips_group_by_user_mode: {trips_group_by_user_mode[mode].keys()}')
        options.append(
            create_single_option(
                mode,
                trips_group_by_user_mode[mode]['color'],
                icon=trips_group_by_user_mode[mode].get('icon')
            )
        )
    return options

def create_ble_modes_dropdown_options(trips_data):
    """
    Creates a list of dropdown options for the "BLE Modes" filter,
    but each option is a 'base mode' (CAR, E_CAR, etc.) rather than
    per-vehicle IDs.
    """
    base_modes = list(baseMode_to_vehicles.keys())  # e.g. ["CAR", "E_CAR", "WHEELCHAIR", ...]
    options = []
    color_palette = ['gray', 'blue', 'green', 'orange', 'purple', 'teal']

    for idx, base_mode in enumerate(base_modes):
        color = color_palette[idx % len(color_palette)]
        options.append(create_single_option(base_mode, color, label=base_mode))
    return options

def create_sensed_modes_dropdown_options(trips_data):
    unique_modes = set()
    for user_data in trips_data.get('users_data_by_user_id', {}).values():
        for trip in user_data.get('trips', []):
            sensed_mode = trip.get("data.primary_sensed_mode")
            if sensed_mode:
                unique_modes.add(sensed_mode)
    options = []
    for mode in sorted(unique_modes):
        options.append(create_single_option(mode, "gray", label=mode))
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
        # Store for label options so it exists
        dcc.Store(id="store-label-options", data={}),
        dcc.Markdown(intro),

        dbc.Row([
            dbc.Col(
                [
                    html.Label('Map Type'),
                    dcc.Dropdown(
                        id='map-type-dropdown',
                        options=map_type_options,
                        value=map_type_options[0]['value'] if map_type_options else None,
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
    labeled_modes_options = create_modes_dropdown_options(
        trips_data.get('users_data_by_user_mode', {})
    )
    filters.append(('Labeled Modes', 'modes', labeled_modes_options))
    logging.info(f'BLE Config: {config}')

    # Conditionally add BLE Modes dropdown if vehicle_identities exist in config
    if config.get('vehicle_identities'):
        ble_modes_options = create_ble_modes_dropdown_options(trips_data)
        filters.append(('BLE Modes', 'ble_modes', ble_modes_options))
    sensed_modes_options = create_sensed_modes_dropdown_options(trips_data)
    filters.append(('Sensed Modes', 'sensed_modes', sensed_modes_options))
    uuids_perm, tokens_perm = has_permission('options_uuids'), has_permission('options_tokens')
    if map_type == 'lines' and (uuids_perm or tokens_perm):
        users_options = create_users_dropdown_options(
            trips_data.get('users_data_by_user_id', {}),
            uuids_perm,
            tokens_perm,
        )
        filters.append(('Users', 'users', users_options))

    return [
        dbc.Col(
            [
                html.Label(label),
                dcc.Dropdown(
                    id={'type': 'map-filter-dropdown', 'id': filter_id},
                    options=options,
                    multi=True,
                    disabled=len(options) == 0,
                    optionHeight=(60 if filter_id == 'users' else 35),
                )
            ],
            lg=4, sm=6,
        )
        for label, filter_id, options in filters
    ]


@callback(
    Output('trip-map-row', 'children'),
    Input('map-type-dropdown', 'value'),
    Input({'type': 'map-filter-dropdown', 'id': ALL}, 'value'),
    Input({'type': 'map-filter-dropdown', 'id': ALL}, 'id'),
    Input('store-trips-map', 'data'),
)
def update_output(map_type, filter_values, filter_ids, trips_data):
    logging.info("=== Entered update_output callback ===")
    logging.info(f"map_type: {map_type}")
    logging.info(f"filter_values: {filter_values}, filter_ids: {filter_ids}")

    # Create a dictionary mapping filter IDs to selected values
    filter_dict = {filter_id['id']: value for filter_id, value in zip(filter_ids, filter_values)}

    selected_modes = filter_dict.get('modes', [])
    selected_ble_modes = filter_dict.get('ble_modes', [])
    selected_sensed_modes = filter_dict.get('sensed_modes', [])
    selected_uuids = filter_dict.get('users', [])

    logging.info(f"selected_modes={selected_modes} | selected_ble_modes={selected_ble_modes} | selected_sensed_modes={selected_sensed_modes} | selected_uuids={selected_uuids}")

    # Apply filters individually, then intersect
    filtered_data = trips_data['users_data_by_user_mode']
    filtered_id_data = trips_data['users_data_by_user_id']

    if selected_ble_modes:
        user_data_by_user_mode_filtered = filter_by_ble_modes(filtered_data, selected_ble_modes)
        user_data_by_user_id_filtered = filter_by_ble_modes(trips_data['users_data_by_user_id'], selected_ble_modes)
    else:
        user_data_by_user_mode_filtered = trips_data['users_data_by_user_mode']
        user_data_by_user_id_filtered = trips_data['users_data_by_user_id']

    if selected_sensed_modes:
        user_data_by_user_mode_filtered = filter_by_sensed_modes(user_data_by_user_mode_filtered, selected_sensed_modes)
        user_data_by_user_id_filtered = filter_by_sensed_modes(user_data_by_user_id_filtered, selected_sensed_modes)

    coordinates = get_map_coordinates(user_data_by_user_mode_filtered, selected_modes)

    # Build the figure based on map_type
    if map_type == 'lines':
        # If user explicitly chose modes, use that; else try user IDs
        if selected_modes:
            logging.info(f"Drawing lines map by labeled modes: {selected_modes}")
            fig = create_lines_map(
                coordinates,
                user_data_by_user_mode_filtered,
                selected_modes or [],
            )
        else:
            logging.info(f"Drawing lines map by user IDs: {selected_uuids}")
            fig = create_lines_map(
                coordinates,
                user_data_by_user_id_filtered,
                selected_uuids or [],
            )
    elif map_type == 'heatmap':
        logging.info("Drawing heatmap")
        fig = create_heatmap_fig(coordinates)
    elif map_type == 'bubble':
        logging.info("Drawing bubble map")
        fig = create_bubble_fig(coordinates)
    else:
        logging.info("No known map_type specified; creating empty figure")
        fig = go.Figure()
    logging.info("=== update_output callback complete ===\n")
    return dcc.Graph(figure=fig)


@callback(
    Output({'type': 'map-filter-dropdown', 'id': 'users'}, 'disabled'),
    Input({'type': 'map-filter-dropdown', 'id': 'modes'}, 'value'),
)
def control_user_dropdowns(selected_modes):
    # If the user has chosen any labeled modes, disable the user selection
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
    # Check if trips_data is None or missing 'data'
    if trips_data is None or not trips_data.get('data'):
        return {'users_data_by_user_id': {}, 'users_data_by_user_mode': {}}
    
    trips_group_by_user_id = get_trips_group_by_user_id(trips_data)
    users_data_by_user_id = process_trips_group(trips_group_by_user_id)
  
    trips_group_by_user_mode = get_trips_group_by_user_mode(trips_data)
    users_data_by_user_mode = process_trips_group(trips_group_by_user_mode)
    # If we have label_options, adjust color/icon for labeled modes
    if label_options:
        rich_modes = [
            [mode, emcdb.get_rich_mode_for_value(mode, label_options)]
            for mode in users_data_by_user_mode
        ]
        # Each element in 'rich_modes' is [mode_string, {"color":..., "icon":...}]

        # Extract the color, then deduplicate
        colors = [[mode, rich_mode['color']] for mode, rich_mode in rich_modes]
        deduped_colors = emcdb.dedupe_colors(colors, adjustment_range=[0.5, 1.5])
        for i, (mode, trips_group) in enumerate(users_data_by_user_mode.items()):
            trips_group['icon'] = rich_modes[i][1]['icon']
            trips_group['color'] = deduped_colors[mode]

    return {
        'users_data_by_user_id': users_data_by_user_id,
        'users_data_by_user_mode': users_data_by_user_mode
    }
