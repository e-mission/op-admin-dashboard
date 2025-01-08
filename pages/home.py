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
import emission.core.timer as ect
import emission.storage.decorations.stats_queries as esdsq

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
    with ect.Timer() as total_timer:

        # Stage 1: Convert 'update_ts' to datetime with UTC
        with ect.Timer() as stage1_timer:
            uuid_df['update_ts'] = pd.to_datetime(uuid_df['update_ts'], utc=True)
        esdsq.store_dashboard_time(
            "admin/home/compute_sign_up_trend/convert_to_datetime",
            stage1_timer
        )

        # Stage 2: Group by date and calculate counts
        with ect.Timer() as stage2_timer:
            res_df = (
                uuid_df
                .groupby(uuid_df['update_ts'].dt.date)
                .size()
                .reset_index(name='count')
                .rename(columns={'update_ts': 'date'})
            )
        esdsq.store_dashboard_time(
            "admin/home/compute_sign_up_trend/group_by_and_calculate_counts",
            stage2_timer
        )

    # Store the total time for the entire function
    esdsq.store_dashboard_time(
        "admin/home/compute_sign_up_trend/total_time",
        total_timer
    )

    return res_df



def compute_trips_trend(trips_df, date_col):
    with ect.Timer() as total_timer:

        # Stage 1: Convert 'date_col' to datetime with UTC
        with ect.Timer() as stage1_timer:
            trips_df[date_col] = pd.to_datetime(trips_df[date_col], utc=True)
        esdsq.store_dashboard_time(
            "admin/home/compute_trips_trend/convert_to_datetime",
            stage1_timer
        )

        # Stage 2: Extract the date part from 'date_col'
        with ect.Timer() as stage2_timer:
            trips_df[date_col] = pd.DatetimeIndex(trips_df[date_col]).date
        esdsq.store_dashboard_time(
            "admin/home/compute_trips_trend/extract_date",
            stage2_timer
        )

        # Stage 3: Group by date and calculate trip counts
        with ect.Timer() as stage3_timer:
            res_df = (
                trips_df
                .groupby(date_col)
                .size()
                .reset_index(name='count')
                .rename(columns={date_col: 'date'})
            )
        esdsq.store_dashboard_time(
            "admin/home/compute_trips_trend/group_by_and_calculate_counts",
            stage3_timer
        )

    # Store the total time for the entire function
    esdsq.store_dashboard_time(
        "admin/home/compute_trips_trend/total_time",
        total_timer
    )

    return res_df



def get_number_of_active_users(uuid_list, threshold):
    with ect.Timer() as total_timer:
        number_of_active_users = 0
        current_timestamp = arrow.utcnow().timestamp()
        for npu in uuid_list:
            user_uuid = UUID(npu)
            profile_data = edb.get_profile_db().find_one({'user_id': user_uuid})
            if profile_data:
                last_call_ts = profile_data.get('last_call_ts')
                if last_call_ts and (current_timestamp - arrow.get(last_call_ts).timestamp()) <= threshold:
                    number_of_active_users += 1
    esdsq.store_dashboard_time("admin/home/get_number_of_active_users/total_time", total_timer)
    return number_of_active_users




def generate_card(title_text, body_text, icon):
    with ect.Timer() as total_timer:

        # Stage 1: Generate the card layout
        with ect.Timer() as stage1_timer:
            card = dbc.CardGroup([
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5(title_text, className="card-title"),
                            html.P(body_text, className="card-text"),
                        ]
                    )
                ),
                dbc.Card(
                    html.Div(className=icon, style=card_icon),
                    className="bg-primary",
                    style={"maxWidth": 75},
                ),
            ])
        esdsq.store_dashboard_time(
            "admin/home/generate_card/generate_card_layout",
            stage1_timer
        )

    # Store the total time for the entire function
    esdsq.store_dashboard_time(
        "admin/home/generate_card/total_time",
        total_timer
    )

    return card


@callback(
    Output('card-users', 'children'),
    Input('store-uuids', 'data'),
)
def update_card_users(store_uuids):
    with ect.Timer() as total_timer:

        # Stage 1: Calculate number of users based on permission
        with ect.Timer() as stage1_timer:
            number_of_users = store_uuids.get('length') if has_permission('overview_users') else 0
        esdsq.store_dashboard_time(
            "update_card_users/calculate_number_of_users",
            stage1_timer
        )

        # Stage 2: Generate the user card
        with ect.Timer() as stage2_timer:
            card = generate_card("# Users", f"{number_of_users} users", "fa fa-users")
        esdsq.store_dashboard_time(
            "update_card_users/generate_user_card",
            stage2_timer
        )

    # Store the total time for the entire function
    esdsq.store_dashboard_time(
        "update_card_users/total_time",
        total_timer
    )

    return card


@callback(
    Output('card-active-users', 'children'),
    Input('store-uuids', 'data'),
)
def update_card_active_users(store_uuids):
    with ect.Timer() as total_timer:

        # Stage 1: Convert store_uuids data to DataFrame
        with ect.Timer() as stage1_timer:
            uuid_df = pd.DataFrame(store_uuids.get('data'))
        esdsq.store_dashboard_time(
            "admin/home/update_card_active_users/convert_to_dataframe",
            stage1_timer
        )

        # Stage 2: Calculate number of active users if DataFrame is not empty and permission is granted
        with ect.Timer() as stage2_timer:
            number_of_active_users = 0
            if not uuid_df.empty and has_permission('overview_active_users'):
                one_day = 24 * 60 * 60
                number_of_active_users = get_number_of_active_users(uuid_df['user_id'], one_day)
        esdsq.store_dashboard_time(
            "admin/home/update_card_active_users/calculate_active_users",
            stage2_timer
        )

        # Stage 3: Generate the active users card
        with ect.Timer() as stage3_timer:
            card = generate_card("# Active users", f"{number_of_active_users} users", "fa fa-person-walking")
        esdsq.store_dashboard_time(
            "admin/home/update_card_active_users/generate_active_users_card",
            stage3_timer
        )

    # Store the total time for the entire function
    esdsq.store_dashboard_time(
        "admin/home/update_card_active_users/total_time",
        total_timer
    )

    return card



@callback(
    Output('card-trips', 'children'),
    Input('store-trips', 'data'),
)
def update_card_trips(store_trips):
    with ect.Timer() as total_timer:

        # Stage 1: Calculate number of trips based on permission
        with ect.Timer() as stage1_timer:
            number_of_trips = store_trips.get('length') if has_permission('overview_trips') else 0
        esdsq.store_dashboard_time(
            "admin/home/update_card_trips/calculate_number_of_trips",
            stage1_timer
        )

        # Stage 2: Generate the trips card
        with ect.Timer() as stage2_timer:
            card = generate_card("# Confirmed trips", f"{number_of_trips} trips", "fa fa-angles-right")
        esdsq.store_dashboard_time(
            "admin/home/update_card_trips/generate_trips_card",
            stage2_timer
        )

    # Store the total time for the entire function
    esdsq.store_dashboard_time(
        "admin/home/update_card_trips/total_time",
        total_timer
    )

    return card



def generate_barplot(data, x, y, title):
    with ect.Timer() as total_timer:

        # Stage 1: Initialize an empty bar plot
        with ect.Timer() as stage1_timer:
            fig = px.bar()
        esdsq.store_dashboard_time(
            "admin/home/generate_barplot/initialize_empty_barplot",
            stage1_timer
        )

        # Stage 2: Generate the bar plot if data is provided
        with ect.Timer() as stage2_timer:
            if data is not None:
                fig = px.bar(data, x=x, y=y)
        esdsq.store_dashboard_time(
            "admin/home/generate_barplot/generate_barplot_with_data",
            stage2_timer
        )

        # Stage 3: Update the layout with the provided title
        with ect.Timer() as stage3_timer:
            fig.update_layout(title=title)
        esdsq.store_dashboard_time(
            "admin/home/generate_barplot/update_layout_with_title",
            stage3_timer
        )

    # Store the total time for the entire function
    esdsq.store_dashboard_time(
        "admin/home/generate_barplot/total_time",
        total_timer
    )

    return fig



@callback(
    Output('fig-sign-up-trend', 'figure'),
    Input('store-uuids', 'data'),
)
def generate_plot_sign_up_trend(store_uuids):
    with ect.Timer() as total_timer:

        # Stage 1: Convert store_uuids data to DataFrame
        with ect.Timer() as stage1_timer:
            df = pd.DataFrame(store_uuids.get("data"))
        esdsq.store_dashboard_time(
            "admin/home/generate_plot_sign_up_trend/convert_to_dataframe",
            stage1_timer
        )

        # Stage 2: Compute the sign-up trend if permission is granted
        with ect.Timer() as stage2_timer:
            trend_df = None
            if not df.empty and has_permission('overview_signup_trends'):
                trend_df = compute_sign_up_trend(df)
        esdsq.store_dashboard_time(
            "admin/home/generate_plot_sign_up_trend/compute_sign_up_trend",
            stage2_timer
        )

        # Stage 3: Generate the bar plot for the sign-up trend
        with ect.Timer() as stage3_timer:
            fig = generate_barplot(trend_df, x='date', y='count', title="Sign-ups trend")
        esdsq.store_dashboard_time(
            "admin/home/generate_plot_sign_up_trend/generate_barplot",
            stage3_timer
        )

    # Store the total time for the entire function
    esdsq.store_dashboard_time(
        "admin/home/generate_plot_sign_up_trend/total_time",
        total_timer
    )

    return fig


@callback(
    Output('fig-trips-trend', 'figure'),
    Input('store-trips', 'data'),
    Input('date-picker', 'start_date'),  # these are ISO strings
    Input('date-picker', 'end_date'),  # these are ISO strings
)
def generate_plot_trips_trend(store_trips, start_date, end_date):
    with ect.Timer() as total_timer:

        # Stage 1: Convert store_trips data to DataFrame
        with ect.Timer() as stage1_timer:
            df = pd.DataFrame(store_trips.get("data"))
        esdsq.store_dashboard_time(
            "admin/home/generate_plot_trips_trend/convert_to_dataframe",
            stage1_timer
        )

        # Stage 2: Convert ISO strings to date-only format
        with ect.Timer() as stage2_timer:
            (start_date, end_date) = iso_to_date_only(start_date, end_date)
        esdsq.store_dashboard_time(
            "admin/home/generate_plot_trips_trend/convert_iso_to_date_only",
            stage2_timer
        )

        # Stage 3: Compute the trips trend if permission is granted
        with ect.Timer() as stage3_timer:
            trend_df = None
            if not df.empty and has_permission('overview_trips_trend'):
                trend_df = compute_trips_trend(df, date_col="trip_start_time_str")
        esdsq.store_dashboard_time(
            "admin/home/generate_plot_trips_trend/compute_trips_trend",
            stage3_timer
        )

        # Stage 4: Generate the bar plot for the trips trend
        with ect.Timer() as stage4_timer:
            fig = generate_barplot(trend_df, x='date', y='count', title=f"Trips trend ({start_date} to {end_date})")
        esdsq.store_dashboard_time(
            "admin/home/generate_plot_trips_trend/generate_barplot",
            stage4_timer
        )

    # Store the total time for the entire function
    esdsq.store_dashboard_time(
        "admin/home/generate_plot_trips_trend/total_time",
        total_timer
    )

    return fig

