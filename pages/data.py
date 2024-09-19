"""
Note that the callback will trigger even if prevent_initial_call=True. This is because dcc.Location must be in app.py.
Since the dcc.Location component is not in the layout when navigating to this page, it triggers the callback.
The workaround is to check if the input value is None.
"""
from dash import dcc, html, Input, Output, callback, register_page, dash_table, State, callback_context, Patch
# Etc
import logging
import time
import pandas as pd
from dash.exceptions import PreventUpdate
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import constants
from utils import permissions as perm_utils
from utils import db_utils
from utils.db_utils import df_to_filtered_records, query_trajectories
from utils.datetime_utils import iso_to_date_only
register_page(__name__, path="/data")

intro = """## Data"""

layout = html.Div(
    [   
        dcc.Markdown(intro),
        dcc.Tabs(id="tabs-datatable", value='tab-uuids-datatable', children=[
            dcc.Tab(label='UUIDs', value='tab-uuids-datatable'),
            dcc.Tab(label='Trips', value='tab-trips-datatable'),
            dcc.Tab(label='Demographics', value='tab-demographics-datatable'),
            dcc.Tab(label='Trajectories', value='tab-trajectories-datatable'),
        ]),
        html.Div(id='tabs-content'),
        dcc.Interval(id='interval-load-more', interval=10000, n_intervals=0), # default loading at 10s, can be lowered or hightened based on perf (usual process local is 3s)
        dcc.Store(id='store-uuids', data=[]),  # Store to hold the original UUIDs data
        dcc.Store(id='store-loaded-uuids', data={'data': [], 'loaded': False})  # Store to track loaded data
    ]
)



def clean_location_data(df):
    if 'data.start_loc.coordinates' in df.columns:
        df['data.start_loc.coordinates'] = df['data.start_loc.coordinates'].apply(lambda x: f'({x[0]}, {x[1]})')
    if 'data.end_loc.coordinates' in df.columns:
        df['data.end_loc.coordinates'] = df['data.end_loc.coordinates'].apply(lambda x: f'({x[0]}, {x[1]})')
    return df

def update_store_trajectories(start_date: str, end_date: str, tz: str, excluded_uuids):
    df = query_trajectories(start_date, end_date, tz)
    records = df_to_filtered_records(df, 'user_id', excluded_uuids["data"])
    store = {
        "data": records,
        "length": len(records),
    }
    return store


@callback(
    Output('tabs-content', 'children'),
    Output('store-loaded-uuids', 'data'),
    Output('interval-load-more', 'disabled'),  # Disable interval when all data is loaded
    Input('tabs-datatable', 'value'),
    Input('store-uuids', 'data'),
    Input('store-excluded-uuids', 'data'),
    Input('store-trips', 'data'),
    Input('store-demographics', 'data'),
    Input('store-trajectories', 'data'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('date-picker-timezone', 'value'),
    Input('interval-load-more', 'n_intervals'),# Interval to trigger the loading of more data
    State('store-loaded-uuids', 'data'),  # Use State to track already loaded data
    State('store-loaded-uuids', 'loaded'),  # Keep track if we have finished loading all data
)
def render_content(tab, store_uuids, store_excluded_uuids, store_trips, store_demographics, store_trajectories, start_date, end_date, timezone, n_intervals, loaded_uuids_store, all_data_loaded):
    initial_batch_size = 10  # Define the batch size for loading UUIDs

    # Ensure store_uuids contains the key 'data' which is a list of dictionaries
    if not isinstance(store_uuids, dict) or 'data' not in store_uuids:
        logging.error(f"Expected store_uuids to be a dict with a 'data' key, but got {type(store_uuids)}")
        return html.Div([html.P("Data structure error.")]), loaded_uuids_store, True

    # Extract the list of UUIDs from the dict
    uuids_list = store_uuids['data']

    # Ensure uuids_list is a list for slicing
    if not isinstance(uuids_list, list):
        logging.error(f"Expected store_uuids['data'] to be a list but got {type(uuids_list)}")
        return html.Div([html.P("Data structure error.")]), loaded_uuids_store, True

    # Retrieve already loaded data from the store
    loaded_data = loaded_uuids_store.get('data', [])
    total_loaded = len(loaded_data)

    # Handle the UUIDs tab with lazy loading
    if tab == 'tab-uuids-datatable' and not loaded_uuids_store.get('loaded', False):
        total_to_load = total_loaded + initial_batch_size
        total_to_load = min(total_to_load, len(uuids_list))  # Avoid loading more than available

        logging.debug(f"Loading next batch of UUIDs: {total_loaded} to {total_to_load}")
        
        # Slice the list of UUIDs from the dict
        new_data = uuids_list[total_loaded:total_to_load]

        if new_data:
            # Process and append the new data to the loaded store
            processed_data = db_utils.add_user_stats(new_data, initial_batch_size)
            loaded_data.extend(processed_data)

            # Update the store with the new data
            loaded_uuids_store['data'] = loaded_data
            loaded_uuids_store['loaded'] = len(loaded_data) >= len(uuids_list)  # Mark all data as loaded if done

            logging.debug(f"New batch loaded. Total loaded: {len(loaded_data)}")

        # Prepare the data to be displayed
        columns = perm_utils.get_uuids_columns()  # Get the relevant columns
        df = pd.DataFrame(loaded_data)

        if df.empty or not perm_utils.has_permission('data_uuids'):
            logging.debug("No data or permission issues.")
            return html.Div([html.P("No data available or you don't have permission.")]), loaded_uuids_store, False

        df = df.drop(columns=[col for col in df.columns if col not in columns])

        logging.debug("Returning appened data to update the UI.")
        return html.Div([populate_datatable(df)]), loaded_uuids_store, False if not loaded_uuids_store['loaded'] else True


    # Handle other tabs normally
    elif tab == 'tab-trips-datatable':
        data = store_trips["data"]
        columns = perm_utils.get_allowed_trip_columns()
        columns.update(col['label'] for col in perm_utils.get_allowed_named_trip_columns())
        columns.update(store_trips["userinputcols"])
        has_perm = perm_utils.has_permission('data_trips')

        df = pd.DataFrame(data)
        if df.empty or not has_perm:
            return None, loaded_uuids_store, True

        df = df.drop(columns=[col for col in df.columns if col not in columns])
        df = clean_location_data(df)

        trips_table = populate_datatable(df, 'trips-table')
        logging.debug(f"Returning 3 values: {trips_table}, {loaded_uuids_store}, True")
        return html.Div([
            html.Button('Display columns with raw units', id='button-clicked', n_clicks=0, style={'marginLeft': '5px'}),
            trips_table
        ]), loaded_uuids_store, True

    elif tab == 'tab-demographics-datatable':
        data = store_demographics["data"]
        has_perm = perm_utils.has_permission('data_demographics')

        if len(data) == 1:
            data = list(data.values())[0]
            columns = list(data[0].keys())
        elif len(data) > 1:
            if not has_perm:
                return None, loaded_uuids_store
            return html.Div([
                dcc.Tabs(id='subtabs-demographics', value=list(data.keys())[0], children=[
                    dcc.Tab(label=key, value=key) for key in data
                ]),
                html.Div(id='subtabs-demographics-content')
            ]), loaded_uuids_store, True

    elif tab == 'tab-trajectories-datatable':
        (start_date, end_date) = iso_to_date_only(start_date, end_date)
        if store_trajectories == {}:
            store_trajectories = update_store_trajectories(start_date, end_date, timezone, store_excluded_uuids)
        data = store_trajectories["data"]
        if data:
            columns = list(data[0].keys())
            columns = perm_utils.get_trajectories_columns(columns)
            has_perm = perm_utils.has_permission('data_trajectories')

        df = pd.DataFrame(data)
        if df.empty or not has_perm:
            return None, loaded_uuids_store

        df = df.drop(columns=[col for col in df.columns if col not in columns])
        return populate_datatable(df), loaded_uuids_store, True

    # Default case: if no data is loaded or the tab is not handled
    return None, loaded_uuids_store, True


# handle subtabs for demographic table when there are multiple surveys
@callback(
    Output('subtabs-demographics-content', 'children'),
    Input('subtabs-demographics', 'value'),
    Input('store-demographics', 'data'),
)

def update_sub_tab(tab, store_demographics):
    data = store_demographics["data"]
    if tab in data:
        data = data[tab]
        if data:
            columns = list(data[0].keys())

        df = pd.DataFrame(data)
        if df.empty:
            return None

        df = df.drop(columns=[col for col in df.columns if col not in columns])

        return populate_datatable(df)


@callback(
    Output('trips-table', 'hidden_columns'), # Output hidden columns in the trips-table
    Output('button-clicked', 'children'), #updates button label
    Input('button-clicked', 'n_clicks'), #number of clicks on the button
    State('button-clicked', 'children') #State representing the current label of button
)
#Controls visibility of columns in trips table  and updates the label of button based on the number of clicks.
def update_dropdowns_trips(n_clicks, button_label):
    if n_clicks % 2 == 0:
        hidden_col = ["data.duration_seconds", "data.distance_meters","data.distance"]
        button_label = 'Display columns with raw units'
    else:
        hidden_col = ["data.duration", "data.distance_miles", "data.distance_km", "data.distance"]
        button_label = 'Display columns with humanzied units'
    #return the list of hidden columns and the updated button label
    return hidden_col, button_label

def populate_datatable(df, table_id=''):
    if not isinstance(df, pd.DataFrame):
        raise PreventUpdate
    return dash_table.DataTable(
        id= table_id,
        # columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        export_format="csv",
        filter_options={"case": "sensitive"},
        # filter_action="native",
        sort_action="native",  # give user capability to sort columns
        sort_mode="single",  # sort across 'multi' or 'single' columns
        page_current=0,  # page number that user is on
        page_size=50,  # number of rows visible per page
        style_cell={
            'textAlign': 'left',
            # 'minWidth': '100px',
            # 'width': '100px',
            # 'maxWidth': '100px',
        },
        style_table={'overflowX': 'auto'},
        css=[{"selector":".show-hide", "rule":"display:none"}]
    )
