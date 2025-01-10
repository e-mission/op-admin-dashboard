"""
Note that the callback will trigger even if prevent_initial_call=True. This is because dcc.Location must be in app.py.
Since the dcc.Location component is not in the layout when navigating to this page, it triggers the callback.
The workaround is to check if the input value is None.
"""
from dash import dcc, html, Input, Output, callback, register_page, State, set_props, dash_table
import dash_ag_grid as dag
# Etc
import logging
import pandas as pd
from dash.exceptions import PreventUpdate

from utils import constants
from utils import permissions as perm_utils
from utils import db_utils
from utils.db_utils import df_to_filtered_records, query_trajectories
from utils.datetime_utils import iso_to_date_only
import emission.core.timer as ect
import emission.storage.decorations.stats_queries as esdsq
from utils.ux_utils import wrap_with_skeleton
register_page(__name__, path="/data")

intro = """## Data"""

layout = html.Div(
    [
        dcc.Markdown(intro),
        html.Div(id='home-page-load', children='', style={'display': 'none'}),
        html.Div(id='card-users'),
        html.Div(id='skeleton-users'),
        html.Div(id='card-active-users'),
        html.Div(id='skeleton-active-users'),
        html.Div(id='card-trips'),
        html.Div(id='skeleton-trips'),
        html.Div(id='fig-sign-up-trend'),
        html.Div(id='skeleton-sign-up-trend'),
        html.Div(id='fig-trips-trend'),
        html.Div(id='skeleton-trips-trend'),
        dcc.Tabs(id="tabs-datatable", value='tab-uuids-datatable', children=[
            dcc.Tab(label='UUIDs', value='tab-uuids-datatable'),
            dcc.Tab(label='Trips', value='tab-trips-datatable'),
            dcc.Tab(label='Demographics', value='tab-demographics-datatable'),
            dcc.Tab(label='Trajectories', value='tab-trajectories-datatable'),
        ]),
        html.Div(id='tabs-content'),
        dcc.Store(id='selected-tab', data='tab-uuids-datatable'),  # Store to hold selected tab
        dcc.Store(id='loaded-uuids-stats', data=[]),
        dcc.Store(id='all-uuids-stats-loaded', data=False),
        dcc.Store(id='uuids-page-current', data=0),  # Store to track current page for UUIDs DataTable
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
    with ect.Timer() as total_timer:

        # Stage 1: Clean start location coordinates
        if 'data.start_loc.coordinates' in df.columns:
            with ect.Timer() as stage1_timer:
                df['data.start_loc.coordinates'] = df['data.start_loc.coordinates'].apply(lambda x: f'({x[0]}, {x[1]})')
            esdsq.store_dashboard_time(
                "admin/data/clean_location_data/clean_start_loc_coordinates",
                stage1_timer
            )

        # Stage 2: Clean end location coordinates
        if 'data.end_loc.coordinates' in df.columns:
            with ect.Timer() as stage2_timer:
                df['data.end_loc.coordinates'] = df['data.end_loc.coordinates'].apply(lambda x: f'({x[0]}, {x[1]})')
            esdsq.store_dashboard_time(
                "admin/data/clean_location_data/clean_end_loc_coordinates",
                stage2_timer
            )

    esdsq.store_dashboard_time(
        "admin/db_utils/clean_location_data/total_time",
        total_timer
    )

    return df

def update_store_trajectories(start_date: str, end_date: str, tz: str, excluded_uuids, key_list):
    with ect.Timer() as total_timer:

        # Stage 1: Query trajectories
        with ect.Timer() as stage1_timer:
            df = query_trajectories(start_date, end_date, tz, key_list)
        esdsq.store_dashboard_time(
            "admin/data/update_store_trajectories/query_trajectories",
            stage1_timer
        )

        # Stage 2: Filter records based on user exclusion
        with ect.Timer() as stage2_timer:
            records = df_to_filtered_records(df, 'user_id', excluded_uuids["data"])
        esdsq.store_dashboard_time(
            "admin/data/update_store_trajectories/filter_records",
            stage2_timer
        )

        # Stage 3: Prepare the store data structure
        with ect.Timer() as stage3_timer:
            store = {
                "data": records,
                "length": len(records),
            }
        esdsq.store_dashboard_time(
            "admin/data/update_store_trajectories/prepare_store_data",
            stage3_timer
        )

    esdsq.store_dashboard_time(
        "admin/data/update_store_trajectories/total_time",
        total_timer
    )

    return store


@callback(
    Output('keylist-switch-container', 'style'),
    Input('tabs-datatable', 'value'),
)
def show_keylist_switch(tab):
    if tab is None:
        raise PreventUpdate
    if tab == 'tab-trajectories-datatable':
        return {'display': 'block'} 
    return {'display': 'none'}  # Hide the keylist-switch on all other tabs


@callback(
    Output('all-uuids-stats-loaded', 'data'),
    Input('tabs-datatable', 'value'),
    Input('store-uuids', 'data'),
    background=True,
    # hide the global spinner while callback is running
    running=[Output('global-loading', 'display'), 'hide', 'auto'],
    # if page changes or tab changes while callback is running, cancel
    cancel=[
        Input('url', 'pathname'),
        Input('tabs-datatable', 'value')
    ],
)
def load_uuids_stats(tab, uuids):
    # Check for None values
    if tab is None or uuids is None:
        raise PreventUpdate

    logging.debug("loading uuids stats for tab %s" % tab)
    if tab != 'tab-uuids-datatable':
        return
    
    # slice uuids into chunks of 10
    uuids_chunks = [uuids['data'][i:i+10] for i in range(0, len(uuids['data']), 10)]
    loaded_stats = []
    for uuids in uuids_chunks:
        processed_uuids = db_utils.add_user_stats(uuids, 10)
        loaded_stats.extend(processed_uuids)
        logging.debug("loaded %s uuids stats: %s" % (len(loaded_stats), loaded_stats))
        set_props('loaded-uuids-stats', {'data': loaded_stats})
    return True


@callback(
    Output('tabs-content', 'children'),
    Input('tabs-datatable', 'value'),
    Input('store-uuids', 'data'),
    Input('store-excluded-uuids', 'data'),
    Input('store-trips', 'data'),
    Input('store-demographics', 'data'),
    Input('store-trajectories', 'data'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('date-picker-timezone', 'value'),
    Input('keylist-switch', 'value'),  # Add keylist-switch to trigger data refresh on change
    Input('uuids-page-current', 'data'),  # Current page number for UUIDs DataTable
    Input('loaded-uuids-stats', 'data'),
)
def render_content(tab, store_uuids, store_excluded_uuids, store_trips, store_demographics, store_trajectories, start_date, end_date, timezone, key_list, current_page, loaded_uuids):
    # Check for None values
    inputs = [tab, store_uuids, store_excluded_uuids, store_trips, store_demographics, store_trajectories,
              start_date, end_date, timezone, key_list, current_page, loaded_uuids]
    if any(val is None for val in inputs):
        raise PreventUpdate

    with ect.Timer() as total_timer:
        initial_batch_size = 10  # Define the batch size for loading UUIDs

        # Stage 1: Update selected tab
        selected_tab = tab
        logging.debug(f"Callback - {selected_tab} Stage 1: Selected tab updated.")

        # Initialize return variables
        content = None

        # Handle the UUIDs tab without fullscreen loading spinner
        if tab == 'tab-uuids-datatable':
            with ect.Timer() as handle_uuids_timer:
                # Prepare the data to be displayed
                columns = perm_utils.get_uuids_columns()  # Get the relevant columns
                df = pd.DataFrame(loaded_uuids)

                if not perm_utils.has_permission('data_uuids'):
                    logging.debug(f"Callback - {selected_tab} insufficient permission.")
                    content = html.Div([html.P("No data available or you don't have permission.")])
                else:
                    if df.empty and len(store_uuids['data']) > 0:
                        logging.debug(f"Callback - {selected_tab} loaded_uuids is empty.")
                        content = html.Div(
                            [wrap_with_skeleton('uuids', 500, html.Div())],
                            style={'margin-top': '36px'}
                        )
                    else:
                        df = df.drop(columns=[col for col in df.columns if col not in columns])
                        logging.debug(f"Callback - {selected_tab} Stage 5: Returning appended data to update the UI.")
                        content = html.Div([
                            populate_datatable(df, store_uuids, table_id='uuid-table'),  # Pass current_page
                            html.P(
                                f"Showing {len(loaded_uuids)} of {len(store_uuids['data'])} UUIDs." +
                                (f" Loading {initial_batch_size} more..." if len(loaded_uuids) < len(store_uuids['data']) else ""),
                                style={'margin': '15px 5px'}
                            )
                        ])

            # Store timing after handling UUIDs tab
            esdsq.store_dashboard_time(
                "admin/data/render_content/handle_uuids_tab",
                handle_uuids_timer
            )

        # Handle Trips tab
        elif tab == 'tab-trips-datatable':
            with ect.Timer() as handle_trips_timer:
                logging.debug(f"Callback - {selected_tab} Stage 2: Handling Trips tab.")

                data = store_trips.get("data", [])
                columns = perm_utils.get_allowed_trip_columns()
                columns.update(col['label'] for col in perm_utils.get_allowed_named_trip_columns())
                columns.update(store_trips.get("userinputcols", []))
                has_perm = perm_utils.has_permission('data_trips')

                df = pd.DataFrame(data)
                if df.empty and has_perm:
                    logging.debug(f"Callback - {selected_tab} loaded_trips is empty.")
                    content = html.Div(
                        [
                            html.Div("No data available", style={'text-align': 'center', 'margin-bottom': '16px'}),
                        ],
                        style={'margin-top': '36px'}
                    )

                elif not has_perm:
                    logging.debug(f"Callback - {selected_tab} Error Stage: No permission or no data available.")
                    content = html.Div([html.P("No data available or you don't have permission.")])
                else:
                    df = df.drop(columns=[col for col in df.columns if col not in columns])
                    df = clean_location_data(df)

                    trips_table = populate_datatable(df, store_uuids, table_id='trips-datatable')

                    content = html.Div([
                        html.Button('Display columns with raw units', id='button-clicked', n_clicks=0, style={'marginLeft': '5px'}),
                        trips_table
                    ])
            # Store timing after handling Trips tab
            esdsq.store_dashboard_time(
                "admin/data/render_content/handle_trips_tab",
                handle_trips_timer
            )

        # Handle Demographics tab
        elif tab == 'tab-demographics-datatable':
            with ect.Timer() as handle_demographics_timer:
                data = store_demographics.get("data", {})
                has_perm = perm_utils.has_permission('data_demographics')

                if len(data) == 1:
                    # Here data is a dictionary
                    data = list(data.values())[0]
                    columns = list(data[0].keys()) if data else []
                    df = pd.DataFrame(data)
                    if df.empty:
                        content = html.Div(
                            [wrap_with_skeleton('demographics', 500, html.Div())],
                            style={'margin-top': '36px'}
                        )
                    else:
                        content = populate_datatable(df, store_uuids)
                elif len(data) > 1:
                    if not has_perm:
                        content = html.Div(
                            [wrap_with_skeleton('demographics', 100, html.Div())],
                            style={'margin-top': '36px'}
                        )
                    else:
                        content = html.Div([
                            dcc.Tabs(id='subtabs-demographics', value=list(data.keys())[0], children=[
                                dcc.Tab(label=key, value=key) for key in data
                            ]),
                            html.Div(id='subtabs-demographics-content')
                        ])
                else:
                    content = None

            # Store timing after handling Demographics tab
            esdsq.store_dashboard_time(
                "admin/data/render_content/handle_demographics_tab",
                handle_demographics_timer
            )

        # Handle Trajectories tab
        elif tab == 'tab-trajectories-datatable':
            # Currently store_trajectories data is loaded only when the respective tab is selected
            # Here we query for trajectory data once "Trajectories" tab is selected
            with ect.Timer() as handle_trajectories_timer:
                (start_date, end_date) = iso_to_date_only(start_date, end_date)
                if store_trajectories == {}:
                    store_trajectories = update_store_trajectories(start_date, end_date, timezone, store_excluded_uuids, key_list)
                data = store_trajectories["data"]
                if data:
                    columns = list(data[0].keys())
                    columns = perm_utils.get_trajectories_columns(columns)
                    has_perm = perm_utils.has_permission('data_trajectories')

                    df = pd.DataFrame(data)
                    if df.empty or not has_perm:
                        logging.debug(f"Callback - {selected_tab} Error Stage: No data available or permission issues.")
                        content = None
                    else:
                        df = df.drop(columns=[col for col in df.columns if col not in columns])

                        datatable = populate_datatable(df, store_uuids)

                        content = datatable
                else:
                    content = None

            # Store timing after handling Trajectories tab
            esdsq.store_dashboard_time(
                "admin/data/render_content/handle_trajectories_tab",
                handle_trajectories_timer
            )

        # Handle unhandled tabs or errors
        else:
            logging.debug(f"Callback - {selected_tab} Error Stage: No data loaded or unhandled tab.")
            content = None

    # Store total timing after all stages
    esdsq.store_dashboard_time(
        "admin/data/render_content/total_time",
        total_timer
    )

    return content

# Handle subtabs for demographic table when there are multiple surveys
@callback(
    Output('subtabs-demographics-content', 'children'),
    Input('subtabs-demographics', 'value'),
    Input('store-demographics', 'data'),
    Input('store-uuids', 'data')
)
def update_sub_tab(tab, store_demographics, store_uuids):
    # Check for None values
    if tab is None or store_demographics is None or store_uuids is None:
        raise PreventUpdate

    with ect.Timer() as total_timer:

        # Stage 1: Retrieve and process data for the selected subtab
        with ect.Timer() as stage1_timer:
            data = store_demographics["data"]
            if tab in data:
                data = data[tab]
                if data:
                    columns = list(data[0].keys())
        esdsq.store_dashboard_time(
            "admin/data/update_sub_tab/retrieve_and_process_data",
            stage1_timer
        )

        # Stage 2: Convert data to DataFrame
        with ect.Timer() as stage2_timer:
            df = pd.DataFrame(data)
            if df.empty:
                esdsq.store_dashboard_time(
                    "admin/data/update_sub_tab/convert_to_dataframe",
                    stage2_timer
                )
                esdsq.store_dashboard_time(
                    "admin/data/update_sub_tab/total_time",
                    total_timer
                )
                return None
        esdsq.store_dashboard_time(
            "admin/data/update_sub_tab/convert_to_dataframe",
            stage2_timer
        )

        # Stage 3: Filter columns based on the allowed set
        with ect.Timer() as stage3_timer:
            df = df.drop(columns=[col for col in df.columns if col not in columns])
        esdsq.store_dashboard_time(
            "admin/data/update_sub_tab/filter_columns",
            stage3_timer
        )

        # Stage 4: Populate the datatable with the cleaned DataFrame
        with ect.Timer() as stage4_timer:
            result = populate_datatable(df, store_uuids)
        esdsq.store_dashboard_time(
            "admin/data/update_sub_tab/populate_datatable",
            stage4_timer
        )

    # Store the total time for the entire function
    esdsq.store_dashboard_time(
        "admin/data/update_sub_tab/total_time",
        total_timer
    )

    return result

@callback(
    Output('trips-datatable', 'hidden_columns'),  # Output hidden columns in the trips-table
    Output('button-clicked', 'children'),  # Updates button label
    Input('button-clicked', 'n_clicks'),  # Number of clicks on the button
    State('button-clicked', 'children')  # State representing the current label of button
)
# Controls visibility of columns in trips table and updates the label of button based on the number of clicks.
def update_dropdowns_trips(n_clicks, button_label):
    if n_clicks is None or button_label is None:
        raise PreventUpdate

    with ect.Timer() as total_timer:

        # Stage 1: Determine hidden columns and button label based on number of clicks
        with ect.Timer() as stage1_timer:
            if n_clicks % 2 == 0:
                hidden_col = ["data.duration_seconds", "data.distance_meters", "data.distance"]
                button_label = 'Display columns with raw units'
            else:
                hidden_col = ["data.duration", "data.distance_miles", "data.distance_km", "data.distance"]
                button_label = 'Display columns with humanized units'
        esdsq.store_dashboard_time(
            "admin/data/update_dropdowns_trips/determine_hidden_columns_and_label",
            stage1_timer
        )

    # Store the total time for the entire function
    esdsq.store_dashboard_time(
        "admin/data/update_dropdowns_trips/total_time",
        total_timer
    )

    # Return the list of hidden columns and the updated button label
    return hidden_col, button_label


def populate_datatable(df, store_uuids, table_id=''):
    with ect.Timer() as total_timer:
        df.fillna("N/A", inplace=True)
        # Stage 1: Check if df is a DataFrame and raise PreventUpdate if not
        with ect.Timer() as stage1_timer:
            if not isinstance(df, pd.DataFrame):
                raise PreventUpdate
        esdsq.store_dashboard_time(
            "admin/data/populate_datatable/check_dataframe_type",
            stage1_timer
        )
        if 'user_token' not in df.columns:
            uuids_df = pd.DataFrame(store_uuids['data'])
            
            # Log UUID DataFrame details
            logging.info(f"UUIDs DF Columns: {uuids_df.columns}")
            logging.info(f"UUIDs DF First Row: {uuids_df.iloc[0].to_dict() if not uuids_df.empty else 'DataFrame is empty'}")

            # Handle missing user_id and map user_token
            df['data.user_id'] = df['data.user_id'].fillna('Unknown')
            uuids_df['user_id'] = uuids_df['user_id'].fillna('Unknown')
            df.fillna("N/A", inplace=True)
            # Map 'user_token' using 'data.user_id' in df and 'user_id' in uuids_df
            logging.info(f'Mapping df[data.user_id] to uuids_df[user_id] for user_token...')
            df['user_token'] = df['data.user_id'].map(uuids_df.set_index('user_id')['user_token']).fillna('Unknown')

            logging.info(f'DataFrame after adding user_token: {df.head()}')
        # Stage 2: Create the DataTable from the DataFrame
        with ect.Timer() as stage2_timer:
            df.fillna("N/A", inplace=True)
            df.columns = [col.replace('.', '_') for col in df.columns]
            # Log data to debug
            logging.info(f"AgGrid rowData: {df.to_dict('records')}")
            logging.info(f"AgGrid columnDefs: {[{'field': col} for col in df.columns]}")
            result = dag.AgGrid(
                id=table_id,
                rowData=df.to_dict('records'),  # Pass data from the DataFrame
                columnDefs=[{"field": col} for col in df.columns],  # Minimal column definitions
                dashGridOptions={
                    "pagination": True,  # Enable basic pagination
                    "paginationPageSize": 50,  # Set page size
                },
            )

        esdsq.store_dashboard_time(
            "admin/data/populate_datatable/create_datatable",
            stage2_timer
        )
        
    esdsq.store_dashboard_time(
        "admin/data/populate_datatable/total_time",
        total_timer
    )
    return result


