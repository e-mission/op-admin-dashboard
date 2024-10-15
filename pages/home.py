"""
Note that the callback will trigger even if prevent_initial_call=True. This is because dcc.Location must
be in app.py.  Since the dcc.Location component is not in the layout when navigating to this page, it triggers the callback.
The workaround is to check if the input value is None.

"""
from uuid import UUID
from dash import dcc, html, Input, Output, callback, register_page, no_update
import dash_bootstrap_components as dbc

import plotly.express as px
import dash_mantine_components as dmc

# Etc
import pandas as pd
import arrow

# e-mission modules
import emission.core.get_database as edb

from utils.permissions import has_permission
from utils.datetime_utils import iso_to_date_only

register_page(__name__, path="/")

intro = "## Home"

card_icon_style = {
    "color": "white",
    "textAlign": "center",
    "fontSize": 30,
    "margin": "auto",
}

def generate_card(title_text, body_text, icon_class):
    return dbc.CardGroup([
        dbc.Card(
            dbc.CardBody([
                html.H5(title_text, className="card-title"),
                html.P(body_text, className="card-text"),
            ])
        ),
        dbc.Card(
            html.I(className=icon_class, style=card_icon_style),  # Font Awesome Icons
            className="bg-primary",
            style={"maxWidth": 75},
        ),
    ])

def generate_barplot(data, x, y, title):
    if data is not None and not data.empty:
        fig = px.bar(data, x=x, y=y)
    else:
        # Create an empty figure with a message
        fig = px.bar(title=title)
        fig.update_layout(
            annotations=[
                dict(
                    text="No data available",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=20)
                )
            ]
        )
    fig.update_layout(title=title)
    return fig

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

def wrap_with_skeleton(component_id, height, children_component):
    return dmc.Skeleton(
        height=height,
        visible=True,
        id=f'skeleton-{component_id}',
        children=children_component
    )

# Define the layout with MantineProvider
layout = dmc.MantineProvider(
    theme={
        "colorScheme": "light",  # or "dark"
    },
    children=html.Div(
        [
            dcc.Markdown(intro),

            # Cards Section
            dbc.Row([
                dbc.Col(
                    wrap_with_skeleton('users', 100, html.Div(id='card-users'))
                ),
                dbc.Col(
                    wrap_with_skeleton('active-users', 100, html.Div(id='card-active-users'))
                ),
                dbc.Col(
                    wrap_with_skeleton('trips', 100, html.Div(id='card-trips'))
                ),
            ], className="mb-4"),  # Add margin-bottom for spacing

            # Plots Section
            dbc.Row([
                dbc.Col(
                    wrap_with_skeleton('sign-up-trend', 300, dcc.Graph(id="fig-sign-up-trend"))
                ),
                dbc.Col(
                    wrap_with_skeleton('trips-trend', 300, dcc.Graph(id="fig-trips-trend"))
                ),
            ]),
        ],
        style={"padding": "20px"}  # Optional padding for aesthetics
    )
)

# Callbacks to update Users Card
@callback(
    Output('card-users', 'children'),
    Output('skeleton-users', 'visible'),
    Input('store-uuids', 'data'),
    Input('url', 'pathname'),
    Input('home-page-load', 'children')
)
def update_card_users(store_uuids, pathname, _):
    if pathname != "/":
        return no_update, no_update

    if store_uuids is None or not has_permission('overview_users'):
        number_of_users = 0
    else:
        number_of_users = store_uuids.get('length', 0)

    card = generate_card("# Users", f"{number_of_users} users", "fa fa-users")
    return card, False  # Hide the skeleton

# Callbacks to update Active Users Card
@callback(
    Output('card-active-users', 'children'),
    Output('skeleton-active-users', 'visible'),
    Input('store-uuids', 'data'),
    Input('url', 'pathname'),
    Input('home-page-load', 'children')
)
def update_card_active_users(store_uuids, pathname, _):
    if pathname != "/":
        return no_update, no_update

    if store_uuids is None:
        uuid_df = pd.DataFrame()
    else:
        uuid_data = store_uuids.get('data', [])
        uuid_df = pd.DataFrame(uuid_data) if isinstance(uuid_data, list) else pd.DataFrame()

    number_of_active_users = 0
    if not uuid_df.empty and has_permission('overview_active_users'):
        one_day = 24 * 60 * 60  # Threshold in seconds
        number_of_active_users = get_number_of_active_users(uuid_df['user_id'], one_day)

    card = generate_card("# Active Users", f"{number_of_active_users} users", "fa fa-person-walking")
    return card, False  # Hide the skeleton

# Callbacks to update Trips Card
@callback(
    Output('card-trips', 'children'),
    Output('skeleton-trips', 'visible'),
    Input('store-trips', 'data'),
    Input('url', 'pathname')
)
def update_card_trips(store_trips, pathname):
    if pathname != "/":
        return no_update, no_update

    if store_trips is None or not has_permission('overview_trips'):
        number_of_trips = 0
    else:
        number_of_trips = store_trips.get('length', 0)

    card = generate_card("# Confirmed Trips", f"{number_of_trips} trips", "fa fa-angles-right")
    return card, False  # Hide the skeleton

# Callbacks to update Sign-Up Trend Graph
@callback(
    Output('fig-sign-up-trend', 'figure'),
    Output('skeleton-sign-up-trend', 'visible'),
    Input('store-uuids', 'data'),
    Input('url', 'pathname')
)
def generate_plot_sign_up_trend(store_uuids, pathname):
    if pathname != "/":
        return no_update, no_update

    if store_uuids is None:
        df = pd.DataFrame()
    else:
        uuid_data = store_uuids.get("data", [])
        df = pd.DataFrame(uuid_data) if isinstance(uuid_data, list) else pd.DataFrame()

    trend_df = None
    if not df.empty and has_permission('overview_signup_trends'):
        trend_df = compute_sign_up_trend(df)

    fig = generate_barplot(trend_df, x='date', y='count', title="Sign-ups Trend")
    return fig, False  # Hide the skeleton

# Callbacks to update Trips Trend Graph
@callback(
    Output('fig-trips-trend', 'figure'),
    Output('skeleton-trips-trend', 'visible'),
    Input('store-trips', 'data'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('url', 'pathname')
)
def generate_plot_trips_trend(store_trips, start_date, end_date, pathname):
    if pathname != "/":
        return no_update, no_update

    if store_trips is None:
        df = pd.DataFrame()
    else:
        trips_data = store_trips.get("data", [])
        df = pd.DataFrame(trips_data) if isinstance(trips_data, list) else pd.DataFrame()

    trend_df = None
    if start_date and end_date:
        start_date, end_date = iso_to_date_only(start_date, end_date)

    if not df.empty and has_permission('overview_trips_trend'):
        trend_df = compute_trips_trend(df, date_col="trip_start_time_str")

    fig = generate_barplot(trend_df, x='date', y='count', title=f"Trips Trend ({start_date} to {end_date})")
    return fig, False  # Hide the skeleton