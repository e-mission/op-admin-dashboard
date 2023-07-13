from uuid import UUID

from dash import dcc, html, Input, Output, State, callback, register_page
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import random

import emission.core.wrapper.user as ecwu
import emission.storage.decorations.user_queries as esdu
import emission.core.get_database as edb
import logging

from utils.db_utils import add_user_stats
from utils.permissions import has_permission


register_page(__name__, path="/user")
intro = """## User Details"""


def get_user_tokens_options():
    uuid_list = esdu.get_all_uuids()
    options = [ecwu.User.fromUUID(uid)._User__email for uid in uuid_list]
    return options


def create_stat_card(title, value):
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


layout = html.Div(
    [
        dcc.Markdown(intro),

        dbc.Row([
            dbc.Col(
                [
                    html.Label('User Token'),
                    dcc.Dropdown(
                        id='user-token-dropdown',
                        options=get_user_tokens_options()
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
            dbc.Col([], width=6)
        ]),
    ]
)


@callback(
    Output('user-stats', 'children'),
    Input('user-token-dropdown', 'value')
)
def update_user_stats(selected_user):
    user_data = {}
    if selected_user is not None:
        logging.info(f"selected user is: {selected_user}")
        user_data = add_user_stats([
            {
                'user_id': str(ecwu.User.fromEmail(selected_user).uuid),
                'token': selected_user,
            }
        ])[0]

    cards = [create_stat_card(title, value) for title, value in user_data.items()]
    return cards


@callback(
    Output('user-trip-map', 'figure'),
    Input('user-token-dropdown', 'value'),
)
def update_output(user_token):
    if user_token is not None:
        user_id = str(ecwu.User.fromEmail(user_token).uuid)
        pass



