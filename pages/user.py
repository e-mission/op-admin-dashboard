import logging

from dash import dcc, html, Input, Output, dash_table, callback, register_page
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

import emission.core.wrapper.user as ecwu

from utils import db_utils
from utils.permissions import has_permission


register_page(__name__, path="/user")
intro = """## User Details"""


def group_trips_daily(trips_df):
    trips_df['end_ts'] = pd.to_datetime(trips_df['end_ts'], unit='s')
    trips_df.set_index('end_ts', inplace=True)
    grouped_df = trips_df.groupby(pd.Grouper(freq='D'))
    return grouped_df


def create_stats_card(title, value):
    card_layout = dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(
                    html.H5(title, className='user-card-title')
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


def create_trips_by_date_table(trips_df):
    grouped_trips = group_trips_daily(trips_df)
    table_data = list()
    for date_time, trips in grouped_trips:
        total_trips = len(trips)
        labeled_trips = (trips['user_input'] != {}).sum()
        table_data.append({
            'date': date_time.date(),
            'total_trips': total_trips,
            'labeled_trips': labeled_trips,
        })

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
                'height': 'auto'
            },
            page_size=20,
        )

    return table


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
            dbc.Col([
                dbc.Row(id='user-stats'),
                dbc.Row(
                    dcc.Graph(id="user-trip-map")
                ),
            ], xl=8, lg=6),
            dbc.Col(html.Div(id='user-table-container'), xl=4, lg=6)
        ]),
    ]
)


@callback(
    Output('user-stats', 'children'),
    Output('user-table-container', 'children'),
    Input('user-token-dropdown', 'value')
)
def update_user_stats(user_token):
    cards = list()
    trips_by_date_table = None
    if user_token is not None:
        user_uuid = ecwu.User.fromEmail(user_token).uuid
        logging.info(f"selected user is: {user_token}")
        user_data = db_utils.add_user_stats([
            {
                'user_id': str(user_uuid),
                'token': user_token,
            }
        ])[0]
        cards = [
            create_stats_card(title, val) for title, val in user_data.items()
        ]

        trips_df = db_utils.get_trips_of_user(user_uuid)
        if len(trips_df) > 0:
            logging.info(f"trips on {trips_df.columns}")
            trips_by_date_table = create_trips_by_date_table(trips_df)

    return cards, trips_by_date_table


@callback(
    Output('user-trip-map', 'figure'),
    Input('user-token-dropdown', 'value'),
)
def update_output(user_token):
    if user_token is not None:
        user_id = str(ecwu.User.fromEmail(user_token).uuid)
        pass



