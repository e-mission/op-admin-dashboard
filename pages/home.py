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
import emission.core.timer as ect
import emission.storage.decorations.stats_queries as esdsq

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
    """
    Computes the sign-up trend by counting the number of unique user sign-ups per day.
    
    :param uuid_df (pd.DataFrame): DataFrame containing user UUIDs with 'update_ts' timestamps.
    :return: pandas DataFrame with columns ['date', 'count'] representing sign-ups per day.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Convert 'update_ts' to datetime
        with ect.Timer() as stage1_timer:
            uuid_df['update_ts'] = pd.to_datetime(uuid_df['update_ts'], utc=True)
        esdsq.store_dashboard_time(
            "home/compute_sign_up_trend/convert_update_ts_to_datetime",
            stage1_timer
        )
        
        # Stage 2: Group by date and count sign-ups
        with ect.Timer() as stage2_timer:
            res_df = (
                uuid_df
                .groupby(uuid_df['update_ts'].dt.date)
                .size()
                .reset_index(name='count')
                .rename(columns={'update_ts': 'date'})
            )
        esdsq.store_dashboard_time(
            "home/compute_sign_up_trend/group_by_date_and_count",
            stage2_timer
        )
    
    esdsq.store_dashboard_time(
        "home/compute_sign_up_trend/total_time",
        total_timer
    )
    
    return res_df


def compute_trips_trend(trips_df, date_col):
    """
    Computes the trips trend by counting the number of trips per specified date column.
    
    :param trips_df (pd.DataFrame): DataFrame containing trip data with a date column.
    :param date_col (str): The column name representing the date to group by.
    :return: pandas DataFrame with columns ['date', 'count'] representing trips per day.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Convert date_col to datetime and extract date
        with ect.Timer() as stage1_timer:
            trips_df[date_col] = pd.to_datetime(trips_df[date_col], utc=True)
            trips_df[date_col] = pd.DatetimeIndex(trips_df[date_col]).date
        esdsq.store_dashboard_time(
            "home/compute_trips_trend/convert_date_col_to_datetime_and_extract_date",
            stage1_timer
        )
        
        # Stage 2: Group by date and count trips
        with ect.Timer() as stage2_timer:
            res_df = (
                trips_df
                .groupby(date_col)
                .size()
                .reset_index(name='count')
                .rename(columns={date_col: 'date'})
            )
        esdsq.store_dashboard_time(
            "home/compute_trips_trend/group_by_date_and_count_trips",
            stage2_timer
        )
    
    esdsq.store_dashboard_time(
        "home/compute_trips_trend/total_time",
        total_timer
    )
    
    return res_df


def find_last_get(uuid_list):
    """
    Finds the last 'POST_/usercache/get' API call timestamp for each user in the provided UUID list.
    
    :param uuid_list (list[str]): List of user UUIDs as strings.
    :return: List of dictionaries with '_id' as user_id and 'write_ts' as the latest timestamp.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Convert UUID strings to UUID objects
        with ect.Timer() as stage1_timer:
            uuid_objects = [UUID(npu) for npu in uuid_list]
        esdsq.store_dashboard_time(
            "home/find_last_get/convert_uuid_strings_to_objects",
            stage1_timer
        )
        
        # Stage 2: Perform aggregate query to find the latest 'write_ts' per user
        with ect.Timer() as stage2_timer:
            pipeline = [
                {'$match': {'user_id': {'$in': uuid_objects}}},
                {'$match': {'metadata.key': 'stats/server_api_time'}},
                {'$match': {'data.name': 'POST_/usercache/get'}},
                {'$group': {'_id': '$user_id', 'write_ts': {'$max': '$metadata.write_ts'}}},
            ]
            last_item = list(edb.get_timeseries_db().aggregate(pipeline))
        esdsq.store_dashboard_time(
            "home/find_last_get/perform_aggregate_query",
            stage2_timer
        )
    
    esdsq.store_dashboard_time(
        "home/find_last_get/total_time",
        total_timer
    )
    
    return last_item


def get_number_of_active_users(uuid_list, threshold):
    """
    Determines the number of active users based on the time threshold since their last 'get' API call.
    
    :param uuid_list (list[str]): List of user UUIDs as strings.
    :param threshold (int): Time threshold in seconds to consider a user as active.
    :return: Integer representing the number of active users.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Find last 'get' API call entries for users
        with ect.Timer() as stage1_timer:
            last_get_entries = find_last_get(uuid_list)
        esdsq.store_dashboard_time(
            "home/get_number_of_active_users/find_last_get_entries",
            stage1_timer
        )
        
        # Stage 2: Calculate number of active users based on threshold
        with ect.Timer() as stage2_timer:
            number_of_active_users = 0
            current_timestamp = arrow.utcnow().timestamp()
            for item in last_get_entries:
                last_get = item.get('write_ts')
                if last_get is not None:
                    last_call_diff = current_timestamp - last_get
                    if last_call_diff <= threshold:
                        number_of_active_users += 1
        esdsq.store_dashboard_time(
            "home/get_number_of_active_users/calculate_active_users",
            stage2_timer
        )
    
    esdsq.store_dashboard_time(
        "home/get_number_of_active_users/total_time",
        total_timer
    )
    
    return number_of_active_users


def generate_card(title_text, body_text, icon):
    """
    Generates a Bootstrap CardGroup with a title, body text, and an icon.
    
    :param title_text (str): The title text for the card.
    :param body_text (str): The body text for the card.
    :param icon (str): The CSS class for the icon to display.
    :return: A Dash Bootstrap CardGroup component.
    """
    with ect.Timer() as total_timer:
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
        "home/generate_card/total_time",
        total_timer
    )
    return card


@callback(
    Output('card-users', 'children'),
    Input('store-uuids', 'data'),
)
def update_card_users(store_uuids):
    """
    Updates the '# Users' card with the number of users.
    
    :param store_uuids (dict): Dictionary containing user UUID data.
    :return: Dash Bootstrap CardGroup component displaying the number of users.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Retrieve Number of Users
        with ect.Timer() as stage1_timer:
            number_of_users = store_uuids.get('length') if has_permission('overview_users') else 0
        esdsq.store_dashboard_time(
            "home/update_card_users/retrieve_number_of_users",
            stage1_timer
        )
        
        # Stage 2: Generate User Card
        with ect.Timer() as stage2_timer:
            card = generate_card("# Users", f"{number_of_users} users", "fa fa-users")
        esdsq.store_dashboard_time(
            "home/update_card_users/generate_user_card",
            stage2_timer
        )
    
    esdsq.store_dashboard_time(
        "home/update_card_users/total_time",
        total_timer
    )
    return card


@callback(
    Output('card-active-users', 'children'),
    Input('store-uuids', 'data'),
)
def update_card_active_users(store_uuids):
    """
    Updates the '# Active users' card with the number of active users.
    
    :param store_uuids (dict): Dictionary containing user UUID data.
    :return: Dash Bootstrap CardGroup component displaying the number of active users.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Create DataFrame from UUID data
        with ect.Timer() as stage1_timer:
            uuid_df = pd.DataFrame(store_uuids.get('data'))
        esdsq.store_dashboard_time(
            "home/update_card_active_users/create_dataframe",
            stage1_timer
        )
        
        # Stage 2: Calculate Number of Active Users
        with ect.Timer() as stage2_timer:
            number_of_active_users = 0
            if not uuid_df.empty and has_permission('overview_active_users'):
                one_day = 24 * 60 * 60
                number_of_active_users = get_number_of_active_users(uuid_df['user_id'], one_day)
        esdsq.store_dashboard_time(
            "home/update_card_active_users/calculate_number_of_active_users",
            stage2_timer
        )
        
        # Stage 3: Generate Active Users Card
        with ect.Timer() as stage3_timer:
            card = generate_card("# Active users", f"{number_of_active_users} users", "fa fa-person-walking")
        esdsq.store_dashboard_time(
            "home/update_card_active_users/generate_active_users_card",
            stage3_timer
        )
    
    esdsq.store_dashboard_time(
        "home/update_card_active_users/total_time",
        total_timer
    )
    return card


@callback(
    Output('card-trips', 'children'),
    Input('store-trips', 'data'),
)
def update_card_trips(store_trips):
    """
    Updates the '# Confirmed trips' card with the number of trips.
    
    :param store_trips (dict): Dictionary containing trip data.
    :return: Dash Bootstrap CardGroup component displaying the number of confirmed trips.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Retrieve Number of Trips
        with ect.Timer() as stage1_timer:
            number_of_trips = store_trips.get('length') if has_permission('overview_trips') else 0
        esdsq.store_dashboard_time(
            "home/update_card_trips/retrieve_number_of_trips",
            stage1_timer
        )
        
        # Stage 2: Generate Trips Card
        with ect.Timer() as stage2_timer:
            card = generate_card("# Confirmed trips", f"{number_of_trips} trips", "fa fa-angles-right")
        esdsq.store_dashboard_time(
            "home/update_card_trips/generate_trips_card",
            stage2_timer
        )
    
    esdsq.store_dashboard_time(
        "home/update_card_trips/total_time",
        total_timer
    )
    return card


def generate_barplot(data, x, y, title):
    """
    Generates a Plotly bar plot based on the provided data.
    
    :param data (pd.DataFrame): The data to plot.
    :param x (str): The column name for the x-axis.
    :param y (str): The column name for the y-axis.
    :param title (str): The title of the plot.
    :return: A Plotly Figure object representing the bar plot.
    """
    with ect.Timer() as total_timer:
        fig = px.bar()
        if data is not None:
            fig = px.bar(data, x=x, y=y)
        fig.update_layout(title=title)
    esdsq.store_dashboard_time(
        "home/generate_barplot/total_time",
        total_timer
    )
    return fig


@callback(
    Output('fig-sign-up-trend', 'figure'),
    Input('store-uuids', 'data'),
)
def generate_plot_sign_up_trend(store_uuids):
    """
    Generates a bar plot showing the sign-up trend over time.
    
    :param store_uuids (dict): Dictionary containing user UUID data.
    :return: Plotly Figure object representing the sign-up trend bar plot.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Convert UUID Data to DataFrame
        with ect.Timer() as stage1_timer:
            df = pd.DataFrame(store_uuids.get("data"))
        esdsq.store_dashboard_time(
            "home/generate_plot_sign_up_trend/convert_uuid_data_to_dataframe",
            stage1_timer
        )
        
        # Stage 2: Compute Sign-Up Trend
        with ect.Timer() as stage2_timer:
            trend_df = None
            if not df.empty and has_permission('overview_signup_trends'):
                trend_df = compute_sign_up_trend(df)
        esdsq.store_dashboard_time(
            "home/generate_plot_sign_up_trend/compute_sign_up_trend",
            stage2_timer
        )
        
        # Stage 3: Generate Bar Plot
        with ect.Timer() as stage3_timer:
            fig = generate_barplot(trend_df, x='date', y='count', title="Sign-ups trend")
        esdsq.store_dashboard_time(
            "home/generate_plot_sign_up_trend/generate_bar_plot",
            stage3_timer
        )
    
    esdsq.store_dashboard_time(
        "home/generate_plot_sign_up_trend/total_time",
        total_timer
    )
    return fig


@callback(
    Output('fig-trips-trend', 'figure'),
    Input('store-trips', 'data'),
    Input('date-picker', 'start_date'),  # these are ISO strings
    Input('date-picker', 'end_date'),    # these are ISO strings
)
def generate_plot_trips_trend(store_trips, start_date, end_date):
    """
    Generates a bar plot showing the trips trend over a specified date range.
    
    :param store_trips (dict): Dictionary containing trip data.
    :param start_date (str): Start date in ISO format.
    :param end_date (str): End date in ISO format.
    :return: Plotly Figure object representing the trips trend bar plot.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Convert Trip Data to DataFrame
        with ect.Timer() as stage1_timer:
            df = pd.DataFrame(store_trips.get("data"))
        esdsq.store_dashboard_time(
            "home/generate_plot_trips_trend/convert_trip_data_to_dataframe",
            stage1_timer
        )
        
        # Stage 2: Convert and Extract Date Range
        with ect.Timer() as stage2_timer:
            (start_date, end_date) = iso_to_date_only(start_date, end_date)
        esdsq.store_dashboard_time(
            "home/generate_plot_trips_trend/convert_and_extract_date_range",
            stage2_timer
        )
        
        # Stage 3: Compute Trips Trend
        with ect.Timer() as stage3_timer:
            trend_df = None
            if not df.empty and has_permission('overview_trips_trend'):
                trend_df = compute_trips_trend(df, date_col="trip_start_time_str")
        esdsq.store_dashboard_time(
            "home/generate_plot_trips_trend/compute_trips_trend",
            stage3_timer
        )
        
        # Stage 4: Generate Bar Plot
        with ect.Timer() as stage4_timer:
            fig = generate_barplot(trend_df, x='date', y='count', title=f"Trips trend({start_date} to {end_date})")
        esdsq.store_dashboard_time(
            "home/generate_plot_trips_trend/generate_bar_plot",
            stage4_timer
        )
    
    esdsq.store_dashboard_time(
        "home/generate_plot_trips_trend/total_time",
        total_timer
    )
    return fig
