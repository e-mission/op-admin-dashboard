# data.py
"""
Note that the callback will trigger even if prevent_initial_call=True. This is because dcc.Location must be in app.py.
Since the dcc.Location component is not in the layout when navigating to this page, it triggers the callback.
The workaround is to check if the input value is None.
"""
from dash import dcc, html, Input, Output, callback, register_page, State, set_props, MATCH, no_update
import dash_ag_grid as dag
import arrow
import logging
import pandas as pd
from dash.exceptions import PreventUpdate


import io
import zipfile
from datetime import datetime, timedelta

import dash_bootstrap_components as dbc


from utils import constants
from utils import permissions as perm_utils
from utils import db_utils
from utils.db_utils import df_to_filtered_records, query_trajectories
from utils.datetime_utils import iso_to_date_only
import emission.core.timer as ect
import emission.storage.decorations.stats_queries as esdsq
from utils.ux_utils import skeleton
from utils.datetime_utils import ts_to_iso
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
        html.Div(id='tabs-content', style={'margin': '12px '}),
        dcc.Store(id='selected-tab', data='tab-uuids-datatable'),  # Store to hold selected tab
        dcc.Store(id='loaded-uuids-stats', data=[]),
        dcc.Store(id='all-uuids-stats-loaded', data=False),
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
    if tab == 'tab-trajectories-datatable':
        return {'display': 'block'} 
    return {'display': 'none'}  # Hide the keylist-switch on all other tabs


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
)
def render_content(tab, store_uuids, store_excluded_uuids, store_trips, store_demographics, store_trajectories, start_date, end_date, timezone, key_list):
    with ect.Timer() as total_timer:
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
                users_df = pd.DataFrame(store_uuids['data'])

                if users_df.empty or not perm_utils.has_permission('data_uuids'):
                    logging.debug(f"Callback - {selected_tab} insufficient permission.")
                    content = html.Div([html.P("No data available or you don't have permission.")])
                else:
                    users_df = users_df[[c for c in columns if c in users_df.columns]]
                    for col in users_df.columns:
                        if col.endswith('_ts'):
                            users_df[col] = users_df[col].apply(ts_to_iso)

                    if 'total_trips' in users_df.columns and 'labeled_trips' in users_df.columns:
                        loc = users_df.columns.get_loc('labeled_trips') + 1
                        pct = (users_df['labeled_trips'] / users_df['total_trips'])
                        users_df.insert(loc, 'labeled_trips_pct', pct.apply(lambda x: f"{x:.1%}"))

                    logging.debug(f"Callback - {selected_tab} Stage 5: Returning appended data to update the UI.")
                    content = html.Div([
                        populate_datatable(users_df, store_uuids, 'uuids'),
                        html.P(f"Showing {len(store_uuids['data'])} UUIDs.",
                                style={'margin': '15px 5px'})
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
                columns.extend(store_trips.get("userinputcols", []))
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

                    trips_table = populate_datatable(df, store_uuids, 'trips')

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
                        content = skeleton(500)
                    else:
                        content = populate_datatable(df, store_uuids, 'demographics')
                elif len(data) > 1:
                    if not has_perm:
                        content = skeleton(100)
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

                        datatable = populate_datatable(df, store_uuids, 'trajectories')

                        content = datatable
                else:
                    content = html.Div(
                        [
                            html.Div("No data available", style={'text-align': 'center', 'margin-bottom': '16px'}),
                        ],
                        style={'margin-top': '36px'}
                    )

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
            result = populate_datatable(df, store_uuids, 'demographics')
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
    Output({'type': 'data_table', 'id': 'trips'}, 'columnDefs'),
    Output('button-clicked', 'children'),  # Updates button label
    Input('button-clicked', 'n_clicks'),  # Number of clicks on the button
    State({'type': 'data_table', 'id': 'trips'}, 'columnDefs'),
)
# Controls visibility of columns in trips table and updates the label of button based on the number of clicks.
def update_raw_vs_humanized_units(n_clicks, columnDefs):
    with ect.Timer() as total_timer:
        humanized_cols = ['data_duration', 'data_distance_miles', 'data_distance_km']
        raw_cols = ['data_duration_seconds', 'data_distance_meters', 'data_distance']

        # Stage 1: Determine hidden columns and button label based on number of clicks
        with ect.Timer() as stage1_timer:
            if n_clicks % 2 == 0:
                columnDefs = [{**col, 'hide': True} if col['field'] in raw_cols
                              else {**col, 'hide': False}
                              for col in columnDefs]
                button_label = 'Display columns with raw units'
            else:
                columnDefs = [{**col, 'hide': True} if col['field'] in humanized_cols
                              else {**col, 'hide': False}
                              for col in columnDefs]
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
    return columnDefs, button_label


def populate_datatable(df, store_uuids, table_id):
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
            user_id_col = 'data.user_id' if 'data.user_id' in df.columns else 'user_id'
            if user_id_col in df.columns:
                user_id_token_map = uuids_df.set_index('user_id')['user_token'].to_dict()
                df.insert(
                    df.columns.get_loc(user_id_col),
                    'user_token',
                    df[user_id_col].map(user_id_token_map)
                )
        # Stage 2: Create the DataTable from the DataFrame
        with ect.Timer() as stage2_timer:
            # Ag Grid does not allow . in column names; replace with _
            # before creating the DataTable
            df.columns = [col.replace('.', '_') for col in df.columns]
            result = html.Div([
              dag.AgGrid(
                id={'type': 'data_table', 'id': table_id},
                rowData=df.to_dict('records'),
                columnDefs=[{"field": i, "headerName": i} for i in df.columns],
                defaultColDef={ "sortable": True, "filter": True },
                columnSize="autoSize",
                dashGridOptions={
                    "pagination": True,
                    "paginationPageSize": 50,
                    "enableCellTextSelection": True,
                },
                style={
                    "--ag-font-family": "monospace",
                    "height": "600px",
                },
              ),
              dcc.Loading(
                  children=html.Button(
                      "Download ZIP",
                      id={"type": "download-zip-btn", "id": table_id},
                  ),
                  type="default",
                  style={"display": "inline-block"}
              ),
              dcc.Download(id={"type": "download-csv-btn", "id": table_id}),
              dcc.Download(id='download-trajectories-zip'),
            ])
        esdsq.store_dashboard_time(
            "admin/data/populate_datatable/create_datatable",
            stage2_timer
        )
        
    esdsq.store_dashboard_time(
        "admin/data/populate_datatable/total_time",
        total_timer
    )
    return result


@callback(
    Output({"type": "download-csv-btn", "id": MATCH}, "data"),
    Input({"type": "download-zip-btn", "id": MATCH}, "n_clicks"),
    State({"type": "data_table", "id": MATCH}, "rowData"),
    State({"type": "data_table", "id": MATCH}, "id"),
    State("date-picker", "start_date"),
    State("date-picker", "end_date"),
    State("date-picker-timezone", "value"),
    State("store-excluded-uuids", "data"),
    State("keylist-switch", "value"),
    prevent_initial_call=True,
)
def export_table_as_csv(n_clicks, table_data, table_id_dict, start_date, end_date, timezone, store_excluded_uuids, key_list):
    if not n_clicks:
        return no_update
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zip_file:
        
        #special handling for trajectories using chunked download
        if table_id_dict['id'] == 'trajectories':
            logging.info("Starting chunked trajectory export...")
            
            #use chunked download to get complete data beyond 250k limit
            from utils.db_utils import query_all_trajectories_chunked
            from utils.datetime_utils import iso_to_date_only
            
            #convert dates for the chunked query
            (start_date_only, end_date_only) = iso_to_date_only(start_date, end_date)
            
            #track daily data and overall statistics
            daily_data = {}
            total_records = 0
            total_chunks = 0
            all_columns = set()
            unique_users = set()
            
            try:
                #process data in chunks to avoid memory issues
                for chunk_df in query_all_trajectories_chunked(
                    start_date_only, end_date_only, timezone, key_list, store_excluded_uuids["data"]
                ):
                    total_chunks += 1
                    chunk_records = len(chunk_df)
                    total_records += chunk_records
                    
                    logging.info(f"Processing chunk {total_chunks} with {chunk_records} records "
                               f"(Total processed: {total_records})")
                    
                    if not chunk_df.empty:
                        #track columns and users
                        all_columns.update(chunk_df.columns)
                        if 'user_id' in chunk_df.columns:
                            unique_users.update(chunk_df['user_id'].unique())
                        
                        #process timestamp column for daily grouping
                        if 'data.ts' in chunk_df.columns:
                            chunk_df['date'] = chunk_df['data.ts'].apply(
                                lambda ts: arrow.get(ts).format('YYYY-MM-DD') if pd.notna(ts) else 'unknown'
                            )
                            
                            #group this chunk's data by date
                            for date, group_df in chunk_df.groupby('date'):
                                if date not in daily_data:
                                    daily_data[date] = []
                                
                                #remove the temporary date column for storage
                                export_df = group_df.drop('date', axis=1)
                                daily_data[date].append(export_df)
                        else:
                            #fallback: add to unknown date if no timestamp
                            if 'unknown' not in daily_data:
                                daily_data['unknown'] = []
                            daily_data['unknown'].append(chunk_df)
                
                logging.info(f"Chunked download complete: {total_records} total records across {total_chunks} chunks")
                
                #create CSV files for each day
                summary_data = []
                for date, daily_chunks in daily_data.items():
                    #combine all chunks for this date
                    day_df = pd.concat(daily_chunks, ignore_index=True) if daily_chunks else pd.DataFrame()
                    
                    if not day_df.empty:
                        csv_content = day_df.to_csv(index=False)
                        filename = f"trajectories_{date}.csv"
                        zip_file.writestr(filename, csv_content)
                        
                        #track daily summary
                        day_users = day_df['user_id'].nunique() if 'user_id' in day_df.columns else 0
                        summary_data.append({
                            'date': date,
                            'trajectory_count': len(day_df),
                            'unique_users': day_users
                        })
                
                #add overall totals to summary
                summary_data.append({
                    'date': 'TOTAL',
                    'trajectory_count': total_records,
                    'unique_users': len(unique_users)
                })
                
                #create daily summary CSV
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    zip_file.writestr("daily_summary.csv", summary_df.to_csv(index=False))
                
                #create overall export summary
                export_summary = {
                    'total_records': total_records,
                    'total_chunks_processed': total_chunks,
                    'total_days': len([d for d in daily_data.keys() if d != 'unknown']),
                    'date_range': f"{start_date} to {end_date}" if start_date and end_date else "All data",
                    'export_timestamp': datetime.now().isoformat(),
                    'key_list': key_list,
                    'chunked_download': True,
                    'columns': ', '.join(sorted(all_columns))
                }
                export_summary_df = pd.DataFrame([export_summary])
                zip_file.writestr("export_summary.csv", export_summary_df.to_csv(index=False))
                
                logging.info(f"Export complete: {total_records} records across {len(daily_data)} days")
                
            except Exception as e:
                logging.error(f"Error during chunked trajectory export: {str(e)}")
                #create error file in the ZIP
                error_info = {
                    'error': str(e),
                    'export_timestamp': datetime.now().isoformat(),
                    'partial_records_processed': total_records,
                    'chunks_processed': total_chunks
                }
                error_df = pd.DataFrame([error_info])
                zip_file.writestr("export_error.csv", error_df.to_csv(index=False))
        
        else:
            #original behavior for non-trajectories data
            if not table_data:
                return no_update
                
            df = pd.DataFrame(table_data)
            csv_content = df.to_csv(index=False)
            filename = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            zip_file.writestr(filename, csv_content)
            
            #add simple summary
            summary_data = {
                'total_records': len(df),
                'date_range': f"{start_date} to {end_date}" if start_date and end_date else "All data",
                'export_timestamp': datetime.now().isoformat(),
                'chunked_download': False,
                'columns': ', '.join(df.columns.tolist())
            }
            summary_df = pd.DataFrame([summary_data])
            zip_file.writestr("export_summary.csv", summary_df.to_csv(index=False))
    
    zip_buffer.seek(0)
    
    # Return the ZIP file for download
    filename = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return dcc.send_bytes(zip_buffer.getvalue(), filename)