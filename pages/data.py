"""
Note that the callback will trigger even if prevent_initial_call=True. This is because dcc.Location must be in app.py.
Since the dcc.Location component is not in the layout when navigating to this page, it triggers the callback.
The workaround is to check if the input value is None.
"""
from dash import dcc, html, Input, Output, callback, register_page, dash_table, State, callback_context, Patch
# Etc
import logging
import pandas as pd
from dash.exceptions import PreventUpdate
from utils import permissions as perm_utils
from utils import db_utils
from utils.db_utils import df_to_filtered_records, query_trajectories
from utils.datetime_utils import iso_to_date_only
import emission.core.timer as ect
import emission.storage.decorations.stats_queries as esdsq

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
        dcc.Store(id='selected-tab', data='tab-uuids-datatable'),  # Store to hold selected tab
        dcc.Interval(id='interval-load-more', interval=20000, n_intervals=0), # default loading at 10s, can be lowered or hightened based on perf (usual process local is 3s)
        dcc.Store(id='store-uuids', data=[]),  # Store to hold the original UUIDs data
        dcc.Store(id='store-loaded-uuids', data={'data': [], 'loaded': False}),  # Store to track loaded data
        # RadioItems for key list switch, wrapped in a div that can hide/show
        html.Div(
            id='keylist-switch-container',
            children=[
                html.Label("Select Key List:"),
                dcc.RadioItems(
                    id='keylist-switch',
                    options=[
                        {'label': 'Analysis/Recreated Location', 'value': 'analysis/recreated_location'},
                        {'label': 'Background/Location', 'value': 'background/location'}
                    ],
                    value='analysis/recreated_location',  # Default value
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                ),
            ],
            style={'display': 'none'}  # Initially hidden, will show only for the "Trajectories" tab
        ),
    ]
)


def clean_location_data(df):
    """
    Cleans the 'data.start_loc.coordinates' and 'data.end_loc.coordinates' columns
    by formatting them as strings in the format '(latitude, longitude)'.
    
    :param df (pd.DataFrame): The DataFrame containing location data.
    :return: The cleaned DataFrame with formatted coordinate strings.
    """
    with ect.Timer(verbose=False) as total_timer:
        # Stage 1: Clean 'data.start_loc.coordinates'
        with ect.Timer(verbose=False) as stage1_timer:
            if 'data.start_loc.coordinates' in df.columns:
                df['data.start_loc.coordinates'] = df['data.start_loc.coordinates'].apply(
                    lambda x: f'({x[0]}, {x[1]})' if isinstance(x, (list, tuple)) and len(x) >= 2 else x
                )
        esdsq.store_dashboard_time(
            "admin/data/clean_location_data/clean_start_loc_coordinates",
            stage1_timer  # Pass the Timer object
        )

        # Stage 2: Clean 'data.end_loc.coordinates'
        with ect.Timer(verbose=False) as stage2_timer:
            if 'data.end_loc.coordinates' in df.columns:
                df['data.end_loc.coordinates'] = df['data.end_loc.coordinates'].apply(
                    lambda x: f'({x[0]}, {x[1]})' if isinstance(x, (list, tuple)) and len(x) >= 2 else x
                )
        esdsq.store_dashboard_time(
            "admin/data/clean_location_data/clean_end_loc_coordinates",
            stage2_timer  # Pass the Timer object
        )
    
    esdsq.store_dashboard_time(
        "admin/data/clean_location_data/total_time",
        total_timer  # Pass the Timer object
    )
    
    return df



def update_store_trajectories(start_date: str, end_date: str, tz: str, excluded_uuids, key_list):
    """
    Queries trajectories within a specified date range and timezone, filters out excluded UUIDs,
    and returns a store dictionary containing the filtered records and their count.

    :param start_date (str): Start date in ISO format.
    :param end_date (str): End date in ISO format.
    :param tz (str): Timezone string.
    :param excluded_uuids (dict): Dictionary containing UUIDs to exclude.
    :param key_list (list[str]): List of keys to filter trajectories.
    :return: Dictionary with 'data' as filtered records and 'length' as the count of records.
    """
    with ect.Timer(verbose=False) as total_timer:
        # Stage 1: Query Trajectories
        with ect.Timer(verbose=False) as stage1_timer:
            df = query_trajectories(start_date=start_date, end_date=end_date, tz=tz, key_list=key_list)
        esdsq.store_dashboard_time(
            "admin/data/update_store_trajectories/query_trajectories",
            stage1_timer  # Pass the Timer object
        )

        # Stage 2: Filter Records
        with ect.Timer(verbose=False) as stage2_timer:
            records = df_to_filtered_records(df, 'user_id', excluded_uuids["data"])
        esdsq.store_dashboard_time(
            "admin/data/update_store_trajectories/filter_records",
            stage2_timer  # Pass the Timer object
        )

        # Stage 3: Create Store Dictionary
        with ect.Timer(verbose=False) as stage3_timer:
            store = {
                "data": records,
                "length": len(records),
            }
        esdsq.store_dashboard_time(
            "admin/data/update_store_trajectories/create_store_dict",
            stage3_timer  # Pass the Timer object
        )
    
    esdsq.store_dashboard_time(
        "admin/data/update_store_trajectories/total_time",
        total_timer  # Pass the Timer object
    )
    
    return store



@callback(
    Output('keylist-switch-container', 'style'),
    Input('tabs-datatable', 'value'),
)
def show_keylist_switch(tab):
    """
    Toggles the visibility of the keylist switch container based on the selected tab.

    :param tab (str): The currently selected tab.
    :return: Dictionary with CSS style to show or hide the container.
    """
    with ect.Timer(verbose=False) as total_timer:
        # Stage 1: Determine Display Style
        with ect.Timer(verbose=False) as stage1_timer:
            if tab == 'tab-trajectories-datatable':
                style = {'display': 'block'} 
            else:
                style = {'display': 'none'}  # Hide the keylist-switch on all other tabs
        esdsq.store_dashboard_time(
            "admin/data/show_keylist_switch/determine_display_style",
            stage1_timer  # Pass the Timer object
        )
    
    esdsq.store_dashboard_time(
        "admin/data/show_keylist_switch/total_time",
        total_timer  # Pass the Timer object
    )
    
    return style



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
    Input('interval-load-more', 'n_intervals'),  # Interval to trigger the loading of more data
    Input('keylist-switch', 'value'),  # Add keylist-switch to trigger data refresh on change
    State('store-loaded-uuids', 'data'),  # Use State to track already loaded data
    State('store-loaded-uuids', 'loaded')  # Keep track if we have finished loading all data
)
def render_content(tab, store_uuids, store_excluded_uuids, store_trips, store_demographics, store_trajectories,
                   start_date, end_date, timezone, n_intervals, key_list, loaded_uuids_store, all_data_loaded):
    initial_batch_size = 10  # Define the batch size for loading UUIDs

    # Update selected tab
    selected_tab = tab
    logging.debug(f"Selected tab: {selected_tab}")
    # Handle the UUIDs tab without fullscreen loading spinner
    if tab == 'tab-uuids-datatable':
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

        # Handle lazy loading
        if not loaded_uuids_store.get('loaded', False):
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
            return html.Div([html.P("No data available or you don't have permission.")]), loaded_uuids_store, True

        df = df.drop(columns=[col for col in df.columns if col not in columns])

        logging.debug("Returning appended data to update the UI.")
        content = html.Div([
            populate_datatable(df),
            html.P(
                f"Showing {len(loaded_data)} of {len(uuids_list)} UUIDs." +
                (f" Loading 10 more..." if not loaded_uuids_store.get('loaded', False) else ""),
                style={'margin': '15px 5px'}
            )
        ])
        return content, loaded_uuids_store, False if not loaded_uuids_store['loaded'] else True

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
                return None, loaded_uuids_store, True
            return html.Div([
                dcc.Tabs(id='subtabs-demographics', value=list(data.keys())[0], children=[
                    dcc.Tab(label=key, value=key) for key in data
                ]),
                html.Div(id='subtabs-demographics-content')
            ]), loaded_uuids_store, True

    elif tab == 'tab-trajectories-datatable':
        (start_date, end_date) = iso_to_date_only(start_date, end_date)

        # Fetch new data based on the selected key_list from the keylist-switch
        if store_trajectories == {} or key_list:  # Ensure data is refreshed when key_list changes
            store_trajectories = update_store_trajectories(start_date, end_date, timezone, store_excluded_uuids, key_list)

        data = store_trajectories.get("data", [])
        if data:
            columns = list(data[0].keys())
            columns = perm_utils.get_trajectories_columns(columns)
            has_perm = perm_utils.has_permission('data_trajectories')

        df = pd.DataFrame(data)
        if df.empty or not has_perm:
            # If no permission or data, disable interval and return empty content
            return None, loaded_uuids_store, True

        # Filter the columns based on permissions
        df = df.drop(columns=[col for col in df.columns if col not in columns])

        # Return the populated DataTable
        return populate_datatable(df), loaded_uuids_store, True

    # Default case: if no data is loaded or the tab is not handled
    return None, loaded_uuids_store, True


@callback(
    Output('subtabs-demographics-content', 'children'),
    Input('subtabs-demographics', 'value'),
    Input('store-demographics', 'data'),
)
def update_sub_tab(tab, store_demographics):
    """
    Updates the content of the demographics subtabs based on the selected tab and stored demographics data.

    :param tab (str): The currently selected subtab.
    :param store_demographics (dict): Dictionary containing demographics data.
    :return: Dash components to populate the subtabs content.
    """
    with ect.Timer(verbose=False) as total_timer:
        # Stage 1: Extract Data for Selected Tab
        with ect.Timer(verbose=False) as stage1_timer:
            data = store_demographics.get("data", {})
            if tab in data:
                data = data[tab]
                if data:
                    columns = list(data[0].keys())
            else:
                data = {}
                columns = []
        esdsq.store_dashboard_time(
            "admin/data/update_sub_tab/extract_data_for_selected_tab",
            stage1_timer  # Pass the Timer object
        )

        # Stage 2: Create DataFrame
        with ect.Timer(verbose=False) as stage2_timer:
            df = pd.DataFrame(data)
        esdsq.store_dashboard_time(
            "admin/data/update_sub_tab/create_dataframe",
            stage2_timer  # Pass the Timer object
        )

        # Stage 3: Check if DataFrame is Empty
        with ect.Timer(verbose=False) as stage3_timer:
            if df.empty:
                return None
        esdsq.store_dashboard_time(
            "admin/data/update_sub_tab/check_if_dataframe_empty",
            stage3_timer  # Pass the Timer object
        )

        # Stage 4: Drop Unnecessary Columns
        with ect.Timer(verbose=False) as stage4_timer:
            df = df.drop(columns=[col for col in df.columns if col not in columns])
        esdsq.store_dashboard_time(
            "admin/data/update_sub_tab/drop_unnecessary_columns",
            stage4_timer  # Pass the Timer object
        )

        # Stage 5: Populate DataTable
        with ect.Timer(verbose=False) as stage5_timer:
            table = populate_datatable(df)
        esdsq.store_dashboard_time(
            "admin/data/update_sub_tab/populate_datatable",
            stage5_timer  # Pass the Timer object
        )

    esdsq.store_dashboard_time(
        "admin/data/update_sub_tab/total_time",
        total_timer  # Pass the Timer object
    )

    return table


@callback(
    Output('trips-table', 'hidden_columns'),  # Output hidden columns in the trips-table
    Output('button-clicked', 'children'),    # Updates button label
    Input('button-clicked', 'n_clicks'),    # Number of clicks on the button
    State('button-clicked', 'children')     # Current label of button
)
def update_dropdowns_trips(n_clicks, button_label):
    """
    Controls the visibility of columns in the trips table and updates the button label based on the number of clicks.

    :param n_clicks (int): Number of times the button has been clicked.
    :param button_label (str): Current label of the button.
    :return: Tuple containing the list of hidden columns and the updated button label.
    """
    with ect.Timer(verbose=False) as total_timer:
        # Stage 1: Determine Hidden Columns and Button Label
        with ect.Timer(verbose=False) as stage1_timer:
            if n_clicks is None:
                n_clicks = 0  # Handle initial state when button hasn't been clicked
            if n_clicks % 2 == 0:
                hidden_col = ["data.duration_seconds", "data.distance_meters", "data.distance"]
                button_label = 'Display columns with raw units'
            else:
                hidden_col = ["data.duration", "data.distance_miles", "data.distance_km", "data.distance"]
                button_label = 'Display columns with humanized units'
        esdsq.store_dashboard_time(
            "admin/data/update_dropdowns_trips/determine_hidden_columns_and_button_label",
            stage1_timer  # Pass the Timer object
        )

    esdsq.store_dashboard_time(
        "admin/data/update_dropdowns_trips/total_time",
        total_timer  # Pass the Timer object
    )

    return hidden_col, button_label


def populate_datatable(df, table_id=''):
    """
    Populates a Dash DataTable with the provided DataFrame.

    :param df (pd.DataFrame): The DataFrame to display in the table.
    :param table_id (str, optional): The ID to assign to the DataTable.
    :return: Dash DataTable component.
    """
    with ect.Timer(verbose=False) as total_timer:
        # Stage 1: Validate DataFrame
        with ect.Timer(verbose=False) as stage1_timer:
            if not isinstance(df, pd.DataFrame):
                raise PreventUpdate
        esdsq.store_dashboard_time(
            "admin/data/populate_datatable/validate_dataframe",
            stage1_timer  # Pass the Timer object
        )

        # Stage 2: Create DataTable
        with ect.Timer(verbose=False) as stage2_timer:
            table = dash_table.DataTable(
                id=table_id,
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                export_format="csv",
                filter_options={"case": "sensitive"},
                sort_action="native",  # Give user capability to sort columns
                sort_mode="single",    # Sort across 'multi' or 'single' columns
                page_current=0,        # Page number that user is on
                page_size=50,          # Number of rows visible per page
                style_cell={
                    'textAlign': 'left',
                },
                style_table={'overflowX': 'auto'},
                css=[{"selector": ".show-hide", "rule": "display:none"}]
            )
        esdsq.store_dashboard_time(
            "admin/data/populate_datatable/create_datatable",
            stage2_timer  # Pass the Timer object
        )

    esdsq.store_dashboard_time(
        "admin/data/populate_datatable/total_time",
        total_timer  # Pass the Timer object
    )

    return table