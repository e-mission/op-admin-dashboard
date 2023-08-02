import logging

import pandas as pd
from dash import dcc, html, Input, Output, dash_table, callback, register_page
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import emission.core.wrapper.user as ecwu

from utils import db_utils
from utils.permissions import has_permission


register_page(__name__, path="/user")
intro = """## User Details"""


def group_dataframe_daily(df):
    df['start_ts'] = pd.to_datetime(df['start_ts'], unit='s')
    grouped_df = df.groupby(pd.Grouper(key='start_ts', freq='D'))
    return grouped_df


def create_stats_card(title, value):
    card_layout = dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(
                    html.H5(title, className='user-card-title'),
                    className='user-card-header'
                ),
                dbc.CardBody(
                    html.H2(f'{value}', className='user-card-value'),
                    className='user-card-body'
                ),
            ],
            className='user-card',
            color='secondary',
            inverse=True
        ),
        xl=4, sm=6
    )
    return card_layout


def create_table_base(table_data):
    table = None
    if table_data:
        table_columns = [
            {'name': col, 'id': col} for col in table_data[0].keys()
        ]
        table = dash_table.DataTable(
            id='user-table',
            columns=table_columns,
            data=table_data,
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            page_size=12,
        )

    return table


def create_trips_by_date_table(grouped_trips):
    table_data = list()
    for date_time, trips in grouped_trips:
        total_trips = len(trips)
        labeled_trips = (trips['user_input'] != {}).sum()
        table_data.append({
            'date': date_time.date(),
            'total_trips': total_trips,
            'labeled_trips': labeled_trips,
        })

    return create_table_base(table_data)


def create_trips_table(trips_df):
    table_data = list()
    for i, trip in trips_df.iterrows():
        table_data.append({
            'id': i + 1,
            'duration': trip['duration'],
            'location': trip['start_local_dt_timezone'],
            'added_activities': len(trip['additions']),
            'has_details': 1 if trip['user_input'] else 0
        })

    return create_table_base(table_data)


def create_places_table(places_df):
    table_data = list()
    for i, place in places_df.iterrows():
        table_data.append({
            'id': i + 1,
            'duration': place['duration'],
            'location': place['enter_local_dt_timezone'],
            'added_activities': len(place['additions']),
        })

    return create_table_base(table_data)


def create_heatmap_fig(trips_df):
    lat = list()
    lon = list()
    for item in trips_df['start_loc']:
        lon.append(item['coordinates'][0])
        lat.append(item['coordinates'][1])

    for item in trips_df['end_loc']:
        lon.append(item['coordinates'][0])
        lat.append(item['coordinates'][1])

    fig = go.Figure()
    fig.add_trace(
        go.Densitymapbox(
            lon=lon,
            lat=lat,
        )
    )
    fig.update_layout(
        mapbox_style='open-street-map',
        mapbox_center_lon=lon[0],
        mapbox_center_lat=lat[0],
        mapbox_zoom=9,
        margin={"r": 0, "t": 50, "l": 30, "b": 30},
        height=500,
    )
    return fig


layout = html.Div(
    [
        dcc.Markdown(intro),

        dbc.Row([
            dbc.Col(
                [
                    html.Label('User Token'),
                    dcc.Dropdown(
                        id='user-token-dropdown',
                        options=db_utils.get_all_tokens()
                    ),
                ],
                style={
                    'display': 'block' if has_permission('options_emails') else 'none'
                }, xl=3, lg=4, sm=6
            )
        ]),

        dbc.Row([
            dbc.Col(dbc.Row(id='user-stats'), xl=8, lg=6),
            dbc.Col(html.Div(id='user-trips-by-date'), xl=4, lg=6),
        ]),

        dbc.Row([
            dbc.Col(dbc.Row(id='user-trips-map'), xl=8, lg=6),
            dbc.Col(
                [
                    dbc.Row([
                        dbc.Col(
                            [
                                html.Label('Select a Date'),
                                dcc.Dropdown(
                                    id='user-date-dropdown',
                                    options=[]
                                ),
                            ],
                            id='user-date-dropdown-container',
                            style={'display': 'none'},
                            width=8
                        )
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(id='user-trips'), lg=6),
                        dbc.Col(html.Div(id='user-places'), lg=6),
                    ]),
                ]
            ),
        ]),
    ]
)


@callback(
    Output('user-stats', 'children'),
    Output('user-trips-by-date', 'children'),
    Output('user-trips-map', 'children'),
    Output('user-date-dropdown-container', 'style'),
    Output('user-date-dropdown', 'options'),
    Input('user-token-dropdown', 'value'),
)
def update_user_stats(user_token):
    stat_cards = list()
    trips_by_date_table = None
    trips_map = None
    date_dropdown_style = {'display': 'none'}
    date_dropdown_options = list()

    if user_token is not None:
        user_uuid = ecwu.User.fromEmail(user_token).uuid
        logging.info(f"selected user is: {user_token}")
        user_data = db_utils.add_user_stats([
            {
                'user_id': str(user_uuid),
                'token': user_token,
            }
        ])[0]
        stat_cards = [
            create_stats_card(title, val) for title, val in user_data.items()
        ]

        trips_df = db_utils.get_trips_of_user(user_uuid)
        if len(trips_df) > 0:
            logging.info(f"trips columns: {trips_df.columns}")
            grouped_trips = group_dataframe_daily(trips_df)
            trips_by_date_table = create_trips_by_date_table(grouped_trips)
            grouped_trips_keys = grouped_trips.groups.keys()
            date_dropdown_options = [dt.date() for dt in grouped_trips_keys]
            date_dropdown_style = {'display': 'block'}
            trips_fig = create_heatmap_fig(trips_df)
            trips_map = dcc.Graph(id="user-trip-map", figure=trips_fig)

    return (
        stat_cards,
        trips_by_date_table,
        trips_map,
        date_dropdown_style,
        date_dropdown_options,
    )
