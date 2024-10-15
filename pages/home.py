from uuid import UUID
from dash import dcc, html, Input, Output, callback, register_page, no_update
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

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

def wrap_with_skeleton(component_id, height, children_component):
    return dmc.Skeleton(
        height=height,
        visible=True,  # Initially visible
        id=f'skeleton-{component_id}',
        children=children_component
    )

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

def generate_barplot(data, x, y, title):
    fig = px.bar()
    if data is not None and not data.empty:
        fig = px.bar(data, x=x, y=y)
    fig.update_layout(title=title)
    return fig

@callback(
    [
        Output('card-users', 'children'),
        Output('skeleton-users', 'visible'),
        Output('card-active-users', 'children'),
        Output('skeleton-active-users', 'visible'),
        Output('card-trips', 'children'),
        Output('skeleton-trips', 'visible'),
        Output('fig-sign-up-trend', 'figure'),
        Output('skeleton-sign-up-trend', 'visible'),
        Output('fig-trips-trend', 'figure'),
        Output('skeleton-trips-trend', 'visible'),
    ],
    [
        Input('store-uuids', 'data'),
        Input('store-trips', 'data'),
        Input('date-picker', 'start_date'),
        Input('date-picker', 'end_date'),
        Input('url', 'pathname'),
        Input('home-page-load', 'children')
    ]
)
def update_all_components(store_uuids, store_trips, start_date, end_date, pathname, _):
    if pathname != "/":
        return [no_update] * 10

    # Initialize all outputs
    card_users = no_update
    skeleton_users = no_update
    card_active_users = no_update
    skeleton_active_users = no_update
    card_trips = no_update
    skeleton_trips = no_update
    fig_sign_up_trend = no_update
    skeleton_sign_up_trend = no_update
    fig_trips_trend = no_update
    skeleton_trips_trend = no_update

    # Flags to check if each component's data is ready
    all_ready = True

    # Update Users Card
    if store_uuids is None or not has_permission('overview_users'):
        number_of_users = 0
    else:
        number_of_users = store_uuids.get('length', 0)
    card_users = generate_card("# Users", f"{number_of_users} users", "fa fa-users")

    # Update Active Users Card
    uuid_df = pd.DataFrame(store_uuids.get('data')) if store_uuids else pd.DataFrame()
    number_of_active_users = 0
    if not uuid_df.empty and has_permission('overview_active_users'):
        one_day = 24 * 60 * 60
        number_of_active_users = get_number_of_active_users(uuid_df['user_id'], one_day)
    card_active_users = generate_card("# Active users", f"{number_of_active_users} users", "fa fa-person-walking")

    # Update Trips Card
    number_of_trips = store_trips.get('length') if (store_trips and has_permission('overview_trips')) else 0
    card_trips = generate_card("# Confirmed trips", f"{number_of_trips} trips", "fa fa-angles-right")

    # Update Sign-Up Trend Figure
    df = pd.DataFrame(store_uuids.get("data")) if store_uuids else pd.DataFrame()
    trend_df = None
    if not df.empty and has_permission('overview_signup_trends'):
        trend_df = compute_sign_up_trend(df)
    fig_sign_up_trend = generate_barplot(trend_df, x='date', y='count', title="Sign-ups trend")

    # Update Trips Trend Figure
    trips_df = pd.DataFrame(store_trips.get("data")) if store_trips else pd.DataFrame()
    trend_trips_df = None
    (start_date, end_date) = iso_to_date_only(start_date, end_date)
    if not trips_df.empty and has_permission('overview_trips_trend'):
        trend_trips_df = compute_trips_trend(trips_df, date_col="trip_start_time_str")
    fig_trips_trend = generate_barplot(trend_trips_df, x='date', y='count', title=f"Trips trend ({start_date} to {end_date})")

    # Set all skeletons to False since all components are updated
    skeleton_users = False
    skeleton_active_users = False
    skeleton_trips = False
    skeleton_sign_up_trend = False
    skeleton_trips_trend = False

    return [
        card_users,
        skeleton_users,
        card_active_users,
        skeleton_active_users,
        card_trips,
        skeleton_trips,
        fig_sign_up_trend,
        skeleton_sign_up_trend,
        fig_trips_trend,
        skeleton_trips_trend,
    ]
