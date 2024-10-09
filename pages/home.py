from uuid import UUID
from dash import dcc, html, Input, Output, callback, register_page
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import arrow
import logging
import time
from functools import wraps

# e-mission modules
import emission.core.get_database as edb
from utils.permissions import has_permission
from utils.datetime_utils import iso_to_date_only

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs will be output to the console
    ]
)
logger = logging.getLogger(__name__)

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Starting '{func.__name__}'")
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.debug(f"Finished '{func.__name__}' in {elapsed_time:.4f} seconds")
    return wrapper

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

@log_execution_time
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

@log_execution_time
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

@log_execution_time
def find_last_get(uuid_list):

    # Do we really need this?
    # Looks like this takes the most time
    # uuid_list = [UUID(npu) for npu in uuid_list]
    
    if isinstance(uuid_list, pd.Series):
        uuid_list = uuid_list.tolist()
    
    # Combined $match stages
    pipeline = [
        {
            '$match': {
                'user_id': {'$in': uuid_list},
                'metadata.key': 'stats/server_api_time',
                'data.name': 'POST_/usercache/get'
            }
        },
        {
            '$group': {
                '_id': '$user_id',
                'write_ts': {'$max': '$metadata.write_ts'}
            }
        }
    ]
    
    
    # maybe try profiling
    last_items = list(edb.get_timeseries_db().aggregate(pipeline))
    
    return last_items


@log_execution_time
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

@log_execution_time
def generate_card(title_text, body_text, icon):
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
    return card

@log_execution_time
def generate_barplot(data, x, y, title):
    fig = px.bar()
    if data is not None:
        fig = px.bar(data, x=x, y=y)
    fig.update_layout(title=title)
    return fig

@callback(
    Output('card-users', 'children'),
    Input('store-uuids', 'data'),
)
@log_execution_time
def update_card_users(store_uuids):
    logger.debug("Callback 'update_card_users' triggered")
    number_of_users = store_uuids.get('length') if has_permission('overview_users') else 0
    card = generate_card("# Users", f"{number_of_users} users", "fa fa-users")
    return card

@callback(
    Output('card-active-users', 'children'),
    Input('store-uuids', 'data'),
)
@log_execution_time
def update_card_active_users(store_uuids):
    logger.debug("Callback 'update_card_active_users' triggered")
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
@log_execution_time
def update_card_trips(store_trips):
    logger.debug("Callback 'update_card_trips' triggered")
    number_of_trips = store_trips.get('length') if has_permission('overview_trips') else 0
    card = generate_card("# Confirmed trips", f"{number_of_trips} trips", "fa fa-angles-right")
    return card

@callback(
    Output('fig-sign-up-trend', 'figure'),
    Input('store-uuids', 'data'),
)
@log_execution_time
def generate_plot_sign_up_trend(store_uuids):
    logger.debug("Callback 'generate_plot_sign_up_trend' triggered")
    df = pd.DataFrame(store_uuids.get("data"))
    trend_df = None
    if not df.empty and has_permission('overview_signup_trends'):
        trend_df = compute_sign_up_trend(df)
    fig = generate_barplot(trend_df, x='date', y='count', title="Sign-ups trend")
    return fig

@callback(
    Output('fig-trips-trend', 'figure'),
    Input('store-trips', 'data'),
    Input('date-picker', 'start_date'),  # these are ISO strings
    Input('date-picker', 'end_date'),    # these are ISO strings
)
@log_execution_time
def generate_plot_trips_trend(store_trips, start_date, end_date):
    if store_trips is None:
        logger.debug("Callback 'generate_plot_trips_trend' triggered with store_trips=None")
        return px.bar()  # Return an empty figure or a placeholder

    logger.debug("Callback 'generate_plot_trips_trend' triggered with valid inputs")
    df = pd.DataFrame(store_trips.get("data"))
    trend_df = None
    (start_date, end_date) = iso_to_date_only(start_date, end_date)
    if not df.empty and has_permission('overview_trips_trend'):
        trend_df = compute_trips_trend(df, date_col="trip_start_time_str")
    fig = generate_barplot(trend_df, x='date', y='count', title=f"Trips trend({start_date} to {end_date})")
    return fig
