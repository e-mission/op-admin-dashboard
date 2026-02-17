# data.py
"""
Note that the callback will trigger even if prevent_initial_call=True. This is because dcc.Location must be in app.py.
Since the dcc.Location component is not in the layout when navigating to this page, it triggers the callback.
The workaround is to check if the input value is None.
"""
from dash import dcc, html, Input, Output, callback, register_page, State, set_props, MATCH
import dash_ag_grid as dag
import dash_mantine_components as dmc
import arrow
import logging
import pandas as pd
from dash.exceptions import PreventUpdate
import plotly.express as px # for the donut and bar charts
import urllib.request # for fetching the survey from github
import xml.dom.minidom as minidom # for organizing and parsing xml files
from utils import constants
from utils import permissions as perm_utils
from utils import db_utils
from utils.db_utils import df_to_filtered_records, query_trajectories
from utils.datetime_utils import iso_to_date_only
import emission.core.timer as ect
import emission.storage.decorations.stats_queries as esdsq
import emission.storage.json_wrappers as esj
from utils.ux_utils import skeleton
from utils.datetime_utils import ts_to_iso
register_page(__name__, path="/data")

intro = """## Data"""

layout = html.Div(
    [
        dcc.Markdown(intro),
        dcc.Tabs(id="tabs-datatable", value='tab-users-datatable', children=[
            dcc.Tab(label='Users', value='tab-users-datatable'),
            dcc.Tab(label='Trips', value='tab-trips-datatable'),
            dcc.Tab(label='Surveys', value='tab-surveys-datatable'),
            dcc.Tab(label='Trajectories', value='tab-trajectories-datatable'),
        ]),
    
        html.Div(id='tabs-content', style={'margin': '12px '}),
        dcc.Store(id='selected-tab', data='tab-users-datatable'),  # Store to hold selected tab
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

def update_store_trajectories(start_date: str, end_date: str, tz: str, excluded_uuids: dict, key_list: str):
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
    Input('store-surveys', 'data'),
    Input('store-trajectories', 'data'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('date-picker-timezone', 'value'),
    Input('keylist-switch', 'value'),  # Add keylist-switch to trigger data refresh on change
)
def render_content(tab, store_uuids, store_excluded_uuids, store_trips, store_surveys, store_trajectories, start_date, end_date, timezone, key_list):
    with ect.Timer() as total_timer:
        # Stage 1: Update selected tab
        selected_tab = tab
        logging.debug(f"Callback - {selected_tab} Stage 1: Selected tab updated.")

        # Initialize return variables
        content = None

        # Handle the UUIDs tab without fullscreen loading spinner
        if tab == 'tab-users-datatable':
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

                    trip_labels_enketo = perm_utils.config.get("survey_info", {}).get("trip-labels") == 'ENKETO'
                    if trip_labels_enketo:
                        def extract_response(x):
                            docs = esj.wrapped_loads(x) \
                                    .get('trip_user_input', {}) \
                                    .get('data', {}) \
                                    .get('jsonDocResponse', {})
                            r = next(iter(docs.values()), {})
                            # return response wtihout unneeded fields
                            return {
                                k: v for k, v in r.items()
                                if k not in ['meta', 'attrid', 'start', 'end']
                                and 'xmlns' not in k
                            }
                        response = df['data.user_input'].apply(extract_response)
                        user_input_cols = pd.json_normalize(response)
                    else:
                        user_input_cols = pd.json_normalize(
                            df['data.user_input'].apply(lambda x: esj.wrapped_loads(x) if x is not None else {})
                        )
                    user_input_cols.columns = [f"data.user_input.{col}" for col in user_input_cols.columns]
                    df = pd.concat([df, user_input_cols], axis=1)

                    trips_table = populate_datatable(df, store_uuids, 'trips')

                    content = html.Div([
                        dmc.Checkbox(
                            label="Include human-friendly units for distance and duration",
                            id="humanize-units",
                            checked=True,
                            style={'margin-bottom': '12px'}
                        ),
                        dmc.Checkbox(
                            label="Expand user_input to separate columns",
                            id="expand-user-input",
                            checked=False,
                            style={'margin-bottom': '12px'}
                        ),
                        trips_table
                    ])
            # Store timing after handling Trips tab
            esdsq.store_dashboard_time(
                "admin/data/render_content/handle_trips_tab",
                handle_trips_timer
            )

        # Handle Surveys tab
        elif tab == 'tab-surveys-datatable':
            with ect.Timer() as handle_surveys_timer:
                data = store_surveys.get("data", {})
                has_perm = perm_utils.has_permission('data_demographics')

                if len(data) >= 1:
                    if not has_perm:
                        content = skeleton(100)
                    else:
                        content = html.Div([
                            dcc.Tabs(id='subtabs-surveys', value=list(data.keys())[0], children=[
                                dcc.Tab(label=key, value=key) for key in data
                            ]),
                            html.Div( # Added chart toggle div here under the sub-tabs 
                                id='chart-toggle-container', 
                                children=[ 
                                    html.Label("Chart Type", style={'margin-right': '10px', 'font-weight': 'bold'}), 
                                    dmc.SegmentedControl( 
                                        id="chart-type-toggle", 
                                        value="donut", 
                                        data=[ 
                                            {"value": "donut", "label": "Donut Charts"}, 
                                            {"value": "bar", "label": "Bar Charts"}, 
                                        ],
                                    ), 
                                ],
                                style={'margin': '12px 0', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'} # Set to flex/center for cleaner look # Modified line
                            ), 
                            html.Div(id='subtabs-surveys-content')
                        ])
                else:
                    content = None

            # Store timing after handling surveys tab
            esdsq.store_dashboard_time(
                "admin/data/render_content/handle_surveys_tab",
                handle_surveys_timer
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

def build_survey_dictionaries(survey_name: str): # added this function to parse XML
        try:
            
            psu_xml_map = {
                "UserProfileSurvey": "https://raw.githubusercontent.com/e-mission/nrel-openpath-deploy-configs/main/survey_resources/walk-study-psu/walkstudypsu-onboarding-v2.xml",
                "TripConfirmSurvey": "https://raw.githubusercontent.com/e-mission/nrel-openpath-deploy-configs/main/survey_resources/walk-study-psu/walkstudypsu-trip-v3.xml"
            }
            
            form_path = psu_xml_map.get(survey_name)
            if not form_path: return {}, {}, [] # Return empty list for categorical fields
            
            print(f"DEBUG: Actually fetching PSU XML for {survey_name} from GitHub...")
            result = urllib.request.urlopen(form_path)
            doc = minidom.parse(result) 
            itext_map = {}
            for text_node in doc.getElementsByTagName("text"):
                text_id = text_node.getAttribute("id")
                v_nodes = text_node.getElementsByTagName("value")
                if v_nodes and v_nodes[0].firstChild:
                    itext_map[text_id] = v_nodes[0].firstChild.data
            option_dict, question_dict = {}, {}
            categorical_fields = [] # list to filter for categorical data per Jack's request
            for tag in ['input', 'select', 'select1']:
                for node in doc.getElementsByTagName(tag):
                    ref = node.getAttribute("ref")
                    if ref:
                        parts = ref.split('/')
                        short_id, full_db_id = parts[-1], ".".join(parts[2:])
                        # because only select1 (single-choice) questions are suitable for these charts
                        if tag == 'select1':  # changed to strictly select1 to exclude multi-select fields 
                            categorical_fields.append(short_id)
                            categorical_fields.append(full_db_id)
                        label_nodes = node.getElementsByTagName("label")
                        if label_nodes:
                            l_ref = label_nodes[0].getAttribute("ref")
                            clean_id = l_ref.replace("jr:itext('", "").replace("')", "")
                            question_text = itext_map.get(clean_id, short_id)
                            question_dict[short_id] = question_text
                            question_dict[full_db_id] = question_text
            return question_dict, option_dict, categorical_fields
        except Exception as e:
            logging.error(f'Error parsing survey XML for {survey_name}: {e}') # more graceful error handling
        return {}, {}, [] # Return empty list for categorical fields

def populate_survey_charts(df: pd.DataFrame, question_map: dict, categorical_fields: list, chart_type: str = 'donut'):
    viz_charts = []
    # filter metadata
    # Updated: Only visualize columns identified as categorical from the XML parser
    survey_cols = [c for c in df.columns if c in categorical_fields]
    
    for col in survey_cols:
        # map internal ids to human text
        display_question = question_map.get(col, col.replace('_', ' '))
        
        # skip open-ended text fields
        if any(x in col.lower() for x in ["other", "explain"]):
            continue

        counts = df[col].value_counts().reset_index()
        counts.columns = ['response', 'count']
        
        if counts.empty:
            continue

        if chart_type == 'bar':
            counts['group'] = "Study Total"
            fig = px.bar(counts, y='group', x='count', color='response', orientation='h', text_auto='.1f')
            fig.update_layout(
                barmode='stack',
                barnorm='percent',
                xaxis_title="Percent (%)",
                yaxis_title=None,
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5) # Moved legend to bottom center horizontally
            )
        else:
            fig = px.pie(counts, values='count', names='response', hole=0.6)
            fig.update_layout(showlegend=True)

        fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))

        viz_charts.append(html.Div([
            html.Div(f"{display_question}", style={'text-align': 'center', 'height': '60px', 'overflow-y': 'auto', 'font-size': '13px', 'font-weight': 'bold'}),
            dcc.Graph(id={'type': 'survey-donut', 'index': col}, figure=fig, config={'displayModeBar': False})
        ], style={'width': '31%', 'display': 'inline-block', 'padding': '5px', 'border': '1px solid #eee', 'margin': '1%'}))
            
    return viz_charts
# Handle subtabs for surveys tab when there are multiple surveys
@callback(
    Output('subtabs-surveys-content', 'children'),
    Input('subtabs-surveys', 'value'),
    Input('store-surveys', 'data'),
    Input('store-uuids', 'data'),
    Input('chart-type-toggle', 'value'), # toggle bar chart
)
def update_sub_tab(tab, store_surveys, store_uuids, chart_type):

   
    with ect.Timer() as total_timer:

        # Stage 1: Retrieve and process data for the selected subtab
        with ect.Timer() as stage1_timer: # records time to fetch data verify it exists and map IDs to human questions
            surveys_data = store_surveys["data"]
            if tab not in surveys_data or not surveys_data[tab]: return None
            data = surveys_data[tab]
            # Updated: Receive the categorical_fields list here
            question_map, option_map, categorical_fields = build_survey_dictionaries(tab)

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

        # Stage 3: Filter columns using the question_map from Stage 1
        with ect.Timer() as stage3_timer:
            # we keep columns that are in our map or are essential metadata
            allowed = list(question_map.keys()) + ['_id', 'user_id', 'user_token', 'ts']
            df = df[[c for c in df.columns if c in allowed]]

        esdsq.store_dashboard_time(
            "admin/data/update_sub_tab/filter_columns",
            stage3_timer
        )

        # Stage 4: Populate the datatable with the cleaned DataFrame
        with ect.Timer() as stage4_timer:
            table_result = populate_datatable(df, store_uuids, 'surveys') # times the table generation and stores it to be displayed alongside the new charts
            
        esdsq.store_dashboard_time(
            "admin/data/update_sub_tab/populate_datatable",
            stage4_timer
        )

        # Updated: Pass categorical_fields to filter out text inputs like ZIP codes
        viz_charts = populate_survey_charts(df, question_map, categorical_fields, chart_type)

    return html.Div([
        dmc.Accordion(children=[
            dmc.AccordionItem([
                dmc.AccordionControl(f"Survey Summary Dashboard: {tab}"),
                dmc.AccordionPanel([
                    html.Div(viz_charts, style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'})
                ])
            ], value="summary-panel")
        ]),
        html.Hr(style={'margin-top': '40px'}),
        table_result 
    ])



@callback(
    Output({'type': 'data_table', 'id': 'trips'}, 'columnDefs'),
    Input('humanize-units', 'checked'),
    Input('expand-user-input', 'checked'),
    State({'type': 'data_table', 'id': 'trips'}, 'columnDefs'),
)
def hide_cols(humanize, expand_user_input, columnDefs):
    humanized_cols = ['data:duration_humanized', 'data:distance_miles', 'data:distance_km']
    newColumnDefs = []
    for col in columnDefs:
        if col['field'] in humanized_cols:
            col['hide'] = not humanize
        elif col['field'] == 'data:user_input':
            col['hide'] = expand_user_input
        elif col['field'].startswith('data:user_input:'):
            col['hide'] = not expand_user_input
            col['headerName'] = col['headerName'].replace('user_input:', '')
        newColumnDefs.append(col)
    return newColumnDefs

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
            # Ag Grid does not allow . in column names; replace with :
            # before creating the DataTable
            df.columns = [col.replace('.', ':') for col in df.columns]
            result = html.Div([
              dag.AgGrid(
                id={'type': 'data_table', 'id': table_id},
                rowData=df.to_dict('records'),
                columnDefs=[{"field": i, "headerName": i.replace('data:', '')} for i in df.columns],
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
              dmc.Button(
                  "Download as CSV",
                  id={"type": "download-csv-btn", "id": table_id},
                  variant='outline',
                  style={'margin-block': '8px'},
              ),
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
    Output({"type": "data_table", "id": MATCH}, "exportDataAsCsv"),
    Output({"type": "download-csv-btn", "id": MATCH}, "csvExportParams"),
    Output({"type": "download-csv-btn", "id": MATCH}, "n_clicks"),
    Input({"type": "download-csv-btn", "id": MATCH}, "n_clicks"),
)
def export_table_as_csv(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    fname = f"openpath-data-{arrow.now().isoformat()}.csv"
    return True, {"fileName": fname}, 0


@callback(
    Output({"type": "download-csv-btn", "id": MATCH}, "children"),
    Input({"type": "data_table", "id": MATCH}, "rowData"),
)
def update_n_rows_download(row_data):
    if row_data is None:
        raise PreventUpdate
    return html.Span([
        html.I(className="fa fa-download mx-2"),
        f"Download {len(row_data)} Rows as CSV"
    ])
