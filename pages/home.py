"""
Note that the callback will trigger even if prevent_initial_call=True. This is because dcc.Location must
be in app.py.  Since the dcc.Location component is not in the layout when navigating to this page, it triggers the callback.
The workaround is to check if the input value is None.

"""
from uuid import UUID
from dash import dcc, html, Input, Output, callback, register_page
import dash_bootstrap_components as dbc

import plotly.express as px

# Etc
import pandas as pd
import arrow

# e-mission modules
import emission.core.get_database as edb

from utils.permissions import has_permission
from utils.datetime_utils import iso_to_date_only

register_page(__name__, path="/")

intro = "## Home"

card_icon = {
    "color": "white",
    "textAlign": "center",
    "fontSize": 30,
    "margin": "auto",
}

layout = html.Div(
    [
        dcc.Markdown(intro),

        # Cards
        dbc.Row([
            dbc.Col(id='card-users'),
            dbc.Col(id='card-active-users'),
            dbc.Col(id='card-trips')
        ]),

        # Plots
        dbc.Row([
            dcc.Graph(id="fig-sign-up-trend"),
            dcc.Graph(id="fig-trips-trend"),
        ])
    ]
)


def compute_sign_up_trend(uuid_df):
    uuid_df['update_ts'] = pd.to_datetime(uuid_df['update_ts'], utc=True)
    res_df = (
        uuid_df
        .groupby(uuid_df['update_ts'].dt.date)
        .size()
        .reset_index(name='count')
        .rename(columns={'update_ts': 'date'})
    )
    return res_df


def compute_trips_trend(trips_df, date_col):
    trips_df[date_col] = pd.to_datetime(trips_df[date_col], utc=True)
    trips_df[date_col] = pd.DatetimeIndex(trips_df[date_col]).date
    res_df = (
        trips_df
        .groupby(date_col)
        .size()
        .reset_index(name='count')
        .rename(columns={date_col: 'date'})
    )
    return res_df


def find_last_get(uuid_list):
    uuid_list = [UUID(npu) for npu in uuid_list]
    last_item = list(edb.get_timeseries_db().aggregate([
        {'$match': {'user_id': {'$in': uuid_list}}},
        {'$match': {'metadata.key': 'stats/server_api_time'}},
        {'$match': {'data.name': 'POST_/usercache/get'}},
        {'$group': {'_id': '$user_id', 'write_ts': {'$max': '$metadata.write_ts'}}},
    ]))
    return last_item


def get_number_of_active_users(uuid_list, threshold):
    last_get_entries = find_last_get(uuid_list)
    number_of_active_users = 0
    for item in last_get_entries:
        last_get = item['write_ts']
        if last_get is not None:
            last_call_diff = arrow.get().timestamp() - last_get
            if last_call_diff <= threshold:
                number_of_active_users += 1
    return number_of_active_users


def generate_card(title_text, body_text, icon):
    card = dbc.CardGroup([
            dbc.Card(
                dbc.CardBody(
                    [
                            html.H5(title_text, className="card-title"),
                            html.P(body_text, className="card-text",),
                        ]
                    )
                ),
                dbc.Card(
                    html.Div(className=icon, style=card_icon),
                    className="bg-primary",
                    style={"maxWidth": 75},
                ),
            ])
    return card


@callback(
    Output('card-users', 'children'),
    Input('store-uuids', 'data'),
)
def update_card_users(store_uuids):
    number_of_users = store_uuids.get('length') if has_permission('overview_users') else 0
    card = generate_card("# Users", f"{number_of_users} users", "fa fa-users")
    return card


@callback(
    Output('card-active-users', 'children'),
    Input('store-uuids', 'data'),
)
def update_card_active_users(store_uuids):
    uuid_df = pd.DataFrame(store_uuids.get('data'))
    number_of_active_users = 0
    if not uuid_df.empty and has_permission('overview_active_users'):
        one_day = 24 * 60 * 60
        number_of_active_users = get_number_of_active_users(uuid_df['user_id'], one_day)
    card = generate_card("# Active users", f"{number_of_active_users} users", "fa fa-person-walking")
    return card


@callback(
    Output('card-trips', 'children'),
    Input('store-trips', 'data'),
)
def update_card_trips(store_trips):
    number_of_trips = store_trips.get('length') if has_permission('overview_trips') else 0
    card = generate_card("# Confirmed trips", f"{number_of_trips} trips", "fa fa-angles-right")
    return card


def generate_barplot(data, x, y, title):
    fig = px.bar()
    if data is not None:
        fig = px.bar(data, x=x, y=y)
    fig.update_layout(title=title)
    return fig


@callback(
    Output('fig-sign-up-trend', 'figure'),
    Input('store-uuids', 'data'),
)
def generate_plot_sign_up_trend(store_uuids):
    df = pd.DataFrame(store_uuids.get("data"))
    trend_df = None
    if not df.empty and has_permission('overview_signup_trends'):
        trend_df = compute_sign_up_trend(df)
    fig = generate_barplot(trend_df, x = 'date', y = 'count', title = "Sign-ups trend")
    return fig


@callback(
    Output('fig-trips-trend', 'figure'),
    Input('store-trips', 'data'),
    Input('date-picker', 'start_date'), # these are ISO strings
    Input('date-picker', 'end_date'), # these are ISO strings
)
def generate_plot_trips_trend(store_trips, start_date, end_date):
    df = pd.DataFrame(store_trips.get("data"))
    trend_df = None
    (start_date, end_date) = iso_to_date_only(start_date, end_date)
    if not df.empty and has_permission('overview_trips_trend'):
        trend_df = compute_trips_trend(df, date_col = "trip_start_time_str")
    fig = generate_barplot(trend_df, x = 'date', y = 'count', title = f"Trips trend({start_date} to {end_date})")
    return fig
