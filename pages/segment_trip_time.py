from dash import dcc, html, Input, Output, State, callback, register_page, dash_table
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import pandas as pd

import logging
import json

from utils.permissions import has_permission, permissions
from utils import db_utils

register_page(__name__, path="/segment_trip_time")

intro = """
## Segment average trip time
This page displays some statistics on average trip duration between two selected zones.

### Usage
Using the polygon or square tools on the maps' menu, draw the start (left map) and end (right map) zones to consider.

Data will then be fetched for trips crossing the start zone and then the end zone.

Here are some tips on how to draw zones:
* Zones shouldn't cover more than one parallel road; otherwise, it is unclear which path the user took.
* A bigger zone will give more results, at the cost of lower accuracy in trip durations (the start point could be anywhere in the zone).
* For exhaustivity, zone length should somewhat match the distance a vehicle can cross at the maximum allowed speed in 30 seconds (sample rate).
* A smaller zone will give more accurate time results, but the number of trips might be too low to be significant.
* Zones can be moved and edited using the Edit layer menu, and they can be deleted with the Delete layer button.
* Please be advised that only the last added zone will be considered on each map. It is thus advised to delete existing zones before creating new ones.
"""


not_enough_data_message = f"""
Not enough segments could be found between endpoints. This means that the number of recorded trips going from start to end point is too low. 
* There could be data, but on an insufficient number of users, breaking anonymity (minimum number of users is currently set to {permissions.get('segment_trip_time_min_users', 0)})
* You could try to increase the zone sizes, or chose different start and end points.
"""

initial_maps_center = [32.7, -96.8]
initial_maps_zoom = 5
layout = html.Div(
    [
        dcc.Store(id='link-trip-time-start', data=json.dumps({"features": []})),
        dcc.Store(id='link-trip-time-end', data=json.dumps({"features": []})),
        dcc.Markdown(intro),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4('Start zone selection'),
                        dl.Map(
                            [
                                dl.TileLayer(), 
                                dl.FeatureGroup([
                                    dl.EditControl(
                                        id="stt-edit-control-start", 
                                        draw=dict(circle=False, marker=False, polyline=False, circlemarker=False)
                                    )
                                ])
                            ],
                            #[dl.TileLayer(), dl.LayerGroup(id='stt-trip-layer-start')],
                            id='stt-trip-map-start',
                            style={'height': '500px'},
                            center=initial_maps_center,
                            zoom=initial_maps_zoom
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        html.H4('End zone selection'),
                        dl.Map(
                            [
                                dl.TileLayer(), 
                                dl.FeatureGroup([
                                    dl.EditControl(
                                        id="stt-edit-control-end", 
                                        draw=dict(circle=False, marker=False, polyline=False, circlemarker=False)
                                    )
                                ])
                            ],
                            id='stt-trip-map-end',
                            style={'height': '500px'},
                            center=initial_maps_center,
                            zoom=initial_maps_zoom
                        ),
                    ]
                ),
            ]
        ),
        dbc.Row(
            html.Div(id='message'),
        ),
    ]
)



@callback(
    Output('link-trip-time-start', 'data'),
    Input('stt-edit-control-start', 'geojson'),
    prevent_initial_call=True,
)
def map_start_draw(geojson):
    return json.dumps(geojson)

@callback(
    Output('link-trip-time-end', 'data'),
    Input('stt-edit-control-end', 'geojson'),
    prevent_initial_call=True,
)
def map_end_draw(geojson):
    return json.dumps(geojson)



def format_duration_df(df, time_column_name='Time sample'):
    df['Median time (minutes)'] = df.duration / 60  # convert seconds in minutes
    df = df.reset_index().rename(
        columns={
            'start_fmt_time': time_column_name,
            'duration': 'Median time (seconds)',
            'section': 'Count',
            'mode': 'Mode',
        }
    )
    if time_column_name in df:
        if 'Mode' in df:
            df = df[
                [
                    'Mode',
                    time_column_name,
                    'Median time (seconds)',
                    'Median time (minutes)',
                    'Count',
                ]
            ]  # reorder cols
        else:
            df = df[
                [
                    time_column_name,
                    'Median time (seconds)',
                    'Median time (minutes)',
                    'Count',
                ]
            ]  # reorder cols
    else:
        df = df[
            ['Mode', 'Median time (seconds)', 'Median time (minutes)', 'Count']
        ]  # reorder cols
    df = df.to_dict('records')  # Format for display
    return df


@callback(
    Output('message', 'children'),
    Input('link-trip-time-start', 'data'),
    Input('link-trip-time-end', 'data'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('date-picker-timezone', 'value'),
    Input('store-excluded-uuids', 'data'),
    prevent_initial_call=True,
)
def generate_content_on_endpoints_change(link_trip_time_start_str, link_trip_time_end_str, start_date, end_date, timezone, excluded_uuids):
    link_trip_time_start = json.loads(link_trip_time_start_str)
    link_trip_time_end = json.loads(link_trip_time_end_str)
    if len(link_trip_time_end["features"]) == 0 or len(link_trip_time_start["features"]) == 0:
        return ''
    # logging.debug("link_trip_time_start: " + str(link_trip_time_start))
    # logging.debug("link_trip_time_end: " + str(link_trip_time_end))

    # Warning: This is a database call, looks here if there is a performance hog.
    # From initial tests, this seems to be performing well, without the need to do geoqueries in memory
    df = db_utils.query_segments_crossing_endpoints(
        link_trip_time_start["features"][len(link_trip_time_start["features"])-1],
        link_trip_time_end["features"][len(link_trip_time_end["features"])-1],
        start_date,
        end_date,
        timezone,
        excluded_uuids["data"]
    )
    total_nb_trips = df.shape[0]
    if total_nb_trips > 0:
        # Warning: Another db call here.
        # In theory, we could load all inferred_section modes in memory at start time, instead of fetching it everytime
        # However, when testing it, the operation is quite heavy on the db and on ram.
        # I opted for querying only sections we're interested in, every time. Page load is still decent, especially when the number of section is low.
        mode_by_section_id = db_utils.query_inferred_sections_modes(
            df[['section', 'user_id']].to_dict('records')
        )
        df['mode'] = df['section'].apply(
            lambda section_id: mode_by_section_id[str(section_id)].name
        )
        median_trip_time = df['duration'].median()
        times = pd.to_datetime(df['start_fmt_time'], errors='coerce', utc=True)
        duration_per_hour = format_duration_df(
            df.groupby(times.dt.hour).agg({'duration': 'median', 'section': 'count'}),
            time_column_name='Hour',
        )
        duration_per_mode = format_duration_df(
            df.groupby('mode').agg({'duration': 'median', 'section': 'count'})
        )
        duration_per_mode_per_hour = format_duration_df(
            df.groupby(['mode', times.dt.hour]).agg(
                {'duration': 'median', 'section': 'count'}
            ),
            time_column_name='Hour',
        )
        duration_per_mode_per_month = format_duration_df(
            df.groupby(['mode', times.dt.month]).agg(
                {'duration': 'median', 'section': 'count'}
            ),
            time_column_name='Month',
        )
        return dbc.Row(
            [
                dbc.Col(
                    [
                        html.Br(),
                        html.H3('Results'),
                        html.Div(
                            f'Computed median segment duration is {median_trip_time} seconds, {total_nb_trips} trips considered'
                        ),
                        html.Br(),
                        html.H4('Median segment duration by mode of transport'),
                        dash_table.DataTable(
                            id='duration_per_mode',
                            data=duration_per_mode,
                            sort_action='native',
                            sort_mode='multi',
                            export_format='csv',
                        ),
                        html.Br(),
                        html.H4(
                            'Median segment duration by hour of the day (UTC)'
                        ),
                        dash_table.DataTable(
                            id='duration_per_hour',
                            data=duration_per_hour,
                            sort_action='native',
                            sort_mode='multi',
                            export_format='csv',
                        ),
                        html.Br(),
                        html.H4(
                            'Median segment duration by mode and hour of the day (UTC)'
                        ),
                        dash_table.DataTable(
                            id='duration_per_mode_per_hour',
                            data=duration_per_mode_per_hour,
                            sort_action='native',
                            sort_mode='multi',
                            export_format='csv',
                        ),
                        html.Br(),
                        html.H4('Median segment duration by mode and month'),
                        dash_table.DataTable(
                            id='duration_per_mode_per_month',
                            data=duration_per_mode_per_month,
                            sort_action='native',
                            sort_mode='multi',
                            export_format='csv',
                        ),
                    ],
                    xs=6,
                ),
                dbc.Col(
                    [
                        html.Br(),
                        html.H3('Trips Data'),
                        dash_table.DataTable(
                            id='trips_data',
                            data=df[
                                ['start_fmt_time', 'end_fmt_time', 'mode', 'duration']
                            ].to_dict('records'),
                            page_size=15,
                            sort_action='native',
                            sort_mode='multi',
                            export_format='csv',
                        ),
                    ],
                    xs=6,
                    style={
                        'display': 'block'
                        if has_permission('segment_trip_time_full_trips')
                        else 'none'
                    },
                ),
            ]
        )
    return [html.H3('Results'), dcc.Markdown(not_enough_data_message)]
