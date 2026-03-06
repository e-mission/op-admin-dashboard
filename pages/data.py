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
        dcc.Store(id='selected-tab', data='tab-users-datatable'), # Store to hold selected tab
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
                    value='analysis/recreated_location', # Default value
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                ),
            ],
            style={'display': 'none'}
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
            records = df_to_filtered_records(df, 'user_id', excluded_uuids.get("data", [])) # Added safe get method to prevent KeyError
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
    return {'display': 'none'} # Hide the keylist-switch on all other tabs

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
                columns = perm_utils.get_uuids_columns() # Get the relevant columns
                users_df = pd.DataFrame(store_uuids.get('data', [])) # Added safe get method to prevent KeyError

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
                    # Restoration of the correct empty state for the Trips tab
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
                        # Map the raw backend keys to the specific dfc-fermata survey names so they are fetched properly
                        survey_label_map = {
                            "UserProfileSurvey": "dfc-onboarding", # Renamed the demographic survey tab
                            "TripConfirmSurvey": "dfc-gas-trip", # Renamed the trip survey tab
                            "DfcGasTrip": "dfc-gas-trip", # Map CamelCase key from DB
                            "DfcEvRoamingTrip": "dfc-ev-roaming-trip", #  Map CamelCase key from DB
                            "DfcEvReturnTrip": "dfc-ev-return-trip" # Map CamelCase key from DB
                        }
                        
                        content = html.Div([
                            dcc.Tabs(id='subtabs-surveys', value=list(data.keys())[0], children=[
                                # Apply the dictionary to set a clean label, keeping the original key as the underlying value
                                dcc.Tab(label=survey_label_map.get(key, key), value=key) for key in data
                            ]),
                            html.Div( 
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
                                style={'margin': '12px 0', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
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

def build_survey_dictionaries(survey_name: str):
    try:
        # Full map of your specific survey resources
        psu_xml_map = {
            "dfc-onboarding": "https://raw.githubusercontent.com/e-mission/op-deployment-configs/main/survey_resources/dfc-fermata/dfc-onboarding-v1.xml",
            "dfc-gas-trip": "https://raw.githubusercontent.com/e-mission/op-deployment-configs/main/survey_resources/dfc-fermata/dfc-gas-trip-v1.xml",
            "dfc-ev-roaming-trip": "https://raw.githubusercontent.com/e-mission/op-deployment-configs/main/survey_resources/dfc-fermata/dfc-ev-roaming-trip-v1.xml",
            "dfc-ev-return-trip": "https://raw.githubusercontent.com/e-mission/op-deployment-configs/main/survey_resources/dfc-fermata/dfc-ev-return-trip-v1.xml",
            "fermata-onboarding": "https://raw.githubusercontent.com/e-mission/op-deployment-configs/main/survey_resources/dfc-fermata/fermata-onboarding-v0.xml",
            "fermata-ev-return-trip": "https://raw.githubusercontent.com/e-mission/op-deployment-configs/main/survey_resources/dfc-fermata/fermata-ev-return-trip-v0.xml",
            "UserProfileSurvey": "https://raw.githubusercontent.com/e-mission/op-deployment-configs/main/survey_resources/dfc-fermata/dfc-onboarding-v1.xml",
            "TripConfirmSurvey": "https://raw.githubusercontent.com/e-mission/op-deployment-configs/main/survey_resources/dfc-fermata/dfc-gas-trip-v1.xml",
            "DfcGasTrip": "https://raw.githubusercontent.com/e-mission/op-deployment-configs/main/survey_resources/dfc-fermata/dfc-gas-trip-v1.xml", 
            "DfcEvRoamingTrip": "https://raw.githubusercontent.com/e-mission/op-deployment-configs/main/survey_resources/dfc-fermata/dfc-ev-roaming-trip-v1.xml",
            "DfcEvReturnTrip": "https://raw.githubusercontent.com/e-mission/op-deployment-configs/main/survey_resources/dfc-fermata/dfc-ev-return-trip-v1.xml" 
        }
        
        form_path = psu_xml_map.get(survey_name)
        if not form_path: return {}, {}, [], []
        
        result = urllib.request.urlopen(form_path)
        doc = minidom.parse(result) 
        
        # Build the itext translation map
        itext_map = {}
        for text_node in doc.getElementsByTagName("text"):
            text_id = text_node.getAttribute("id")
            v_nodes = text_node.getElementsByTagName("value")
            if v_nodes and v_nodes[0].firstChild:
                itext_map[text_id] = v_nodes[0].firstChild.data

        option_dict, question_dict = {}, {}
        categorical_fields, multi_select_fields = [], []

        # Logic for translating raw data values (like 1, 2, 3) to human labels
        for item in doc.getElementsByTagName("item"): 
            val_nodes = item.getElementsByTagName("value")
            lab_nodes = item.getElementsByTagName("label")
            if val_nodes and lab_nodes and val_nodes[0].firstChild: # Safety check for empty values
                raw_val = val_nodes[0].firstChild.data
                l_ref = lab_nodes[0].getAttribute("ref")
                if l_ref and "jr:itext" in l_ref: #  Check if it uses a translation reference
                    clean_id = l_ref.replace("jr:itext('", "").replace("')", "")
                    option_dict[raw_val] = itext_map.get(clean_id, raw_val)
                elif lab_nodes[0].firstChild: # Fallback to pull direct text from the label
                    option_dict[raw_val] = lab_nodes[0].firstChild.data.strip()

        # Logic for identifying and labeling questions
        for tag in ['input', 'select', 'select1']:
            for node in doc.getElementsByTagName(tag):
                ref = node.getAttribute("ref")
                if ref:
                    short_id = ref.split('/')[-1]
                    if tag in ['select', 'select1']:
                        categorical_fields.append(short_id)
                    if tag == 'select':
                        multi_select_fields.append(short_id)
                    
                    lab_node = node.getElementsByTagName("label")
                    if lab_node:
                        l_ref = lab_node[0].getAttribute("ref")
                        if l_ref and "jr:itext" in l_ref: # ADDED: Check if it uses a translation reference
                            clean_id = l_ref.replace("jr:itext('", "").replace("')", "")
                            question_dict[short_id] = itext_map.get(clean_id, short_id)
                        elif lab_node[0].firstChild: # ADDED: Fallback to pull direct text from the label
                            question_dict[short_id] = lab_node[0].firstChild.data.strip()

        return question_dict, option_dict, categorical_fields, multi_select_fields
    except Exception as e:
        logging.error(f'Error parsing survey XML for {survey_name}: {e}') 
    return {}, {}, [], []

def populate_survey_charts(df: pd.DataFrame, question_map: dict, option_map: dict, categorical_fields: list, multi_select_fields: list, chart_type: str = 'donut'): 
    viz_charts = []
    
  
    def format_label(val):
        val_str = str(val) # ADDED: Convert value to string for consistent checking
        label = option_map.get(val, option_map.get(val_str, val))
        if val_str.isdigit(): # ADDED: Only apply the Likert formatting if the raw value is a number
            if str(label) == val_str or label == "-":
                return val_str
            return f"{val_str} ({label})"
        return str(label) # ADDED: For all text based questions just return the clean human readable label

    # filter metadata
    # Only visualize columns identified as categorical from the XML parser
    survey_cols = [c for c in df.columns if c in categorical_fields]
    
    for col in survey_cols:
        # map internal ids to human text
        display_question = question_map.get(col, col.replace('_', ' '))
        question_type_label = "" # ADDED: Initialize our new label variable
        
        # skip open-ended text fields
        if any(x in col.lower() for x in ["other", "explain"]):
            continue

        if col in multi_select_fields: 
            question_type_label = "Multi-select" # ADDED: Tag the multiple choice fields
            if chart_type == 'donut': 
                continue 
            total_respondents = len(df[col].dropna())
            # Splitting multi-select strings so each choice is counted individually
            counts = df[col].dropna().astype(str).str.split().explode().value_counts().reset_index()
            counts.columns = ['response', 'count']
            
            if counts.empty:
                continue
            
           
            counts['response'] = counts['response'].apply(lambda x: option_map.get(x, option_map.get(str(x), x))) 
                
            counts['percent'] = (counts['count'] / total_respondents) * 100
            counts = counts.sort_values('percent', ascending=True)
            
            multi_select_colors = ["#B71C1C", "#E65100", "#FF9800", "#FBC02D", "#FFF176"] # ADDED: Switched multiple choice to a strictly warm orange and yellow palette
        
            fig = px.bar(counts, y='response', x='percent', orientation='h', color=counts['response'].astype(str), color_discrete_sequence=multi_select_colors) # ADDED: Applied the strict warm palette
            fig.update_traces(
                texttemplate='%{x:.0f}%',
                textposition='outside',
                cliponaxis=False
                # ADDED: Removed showlegend=False so legend can actually render
            )
            fig.update_layout(
                xaxis_title=None,
                yaxis_title=None,
                showlegend=True, # ADDED: Fixed the layout typo that was crashing the application
                legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5), # ADDED: Push the legend cleanly to the bottom
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, type='category') # ADDED: Hide the repetitive y axis labels since we now have a legend
            )
                
        else:
            # Standard counting for single-choice questions to keep the visualization focused 
            counts = df[col].value_counts().reset_index()
            counts.columns = ['response', 'count']
            
            if counts.empty:
                continue
            
            counts = counts.sort_values(by='response', ascending=True) # ADDED: Sort by the raw numerical value to guarantee proper Likert order

            is_likert = counts['response'].astype(str).str.isdigit().all() # ADDED: Identify if every response is a number to confirm it is a Likert scale
            
            if is_likert and chart_type == 'donut': # ADDED: Force the loop to skip rendering this chart if it is a Likert scale in donut mode
                continue
                
            question_type_label = "Likert" if is_likert else "Single-select" # ADDED: Tag the question properly based on the math check

            counts['response'] = counts['response'].apply(format_label) # ADDED: Apply our new strict formatting function
            ordered_labels = counts['response'].tolist() # ADDED: Extract the exact text order to force the graph hierarchy

            likert_colors = ["#2071ec", "#59b1fd", "#e0e0e0", "#ffb950", "#e39d06"] # ADDED: Reverted back to the original blue and orange diverging scale
            single_select_colors = ["#EE6363", "#6C69FE", "#F7B299", "#607D8B", "#4A148C"] # ADDED: Reverted back to the original high contrast palette for single choice
            color_palette = likert_colors if is_likert else single_select_colors # ADDED: Applied the strict color separation logic
            legend_layout = dict(orientation="v", yanchor="auto", y=0.5, xanchor="left", x=1.02) if is_likert else dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5)

            if chart_type == 'bar':
                counts['group'] = "Study Total"
                
                if is_likert:
                    # ADDED: We must map color to a pure numeric value to force the Plotly Colorbar to generate
                    counts['numeric_val'] = counts['response'].str.extract(r'(\d+)', expand=False).astype(float)
                    
                    # ADDED: We build a stepped 'blocky' colorscale so it acts like distinct blocks instead of a blurry gradient
                    blocky_colorscale = [
                        [0.0, likert_colors[0]], [0.2, likert_colors[0]],
                        [0.2, likert_colors[1]], [0.4, likert_colors[1]],
                        [0.4, likert_colors[2]], [0.6, likert_colors[2]],
                        [0.6, likert_colors[3]], [0.8, likert_colors[3]],
                        [0.8, likert_colors[4]], [1.0, likert_colors[4]]
                    ]
                    
                    fig = px.bar(
                        counts, 
                        y='group', 
                        x='count', 
                        color='numeric_val', # Triggers the colorbar
                        orientation='h', 
                        text_auto='.1f',
                        color_continuous_scale=blocky_colorscale,
                        range_color=[0.5, 5.5] # Centers the 5 blocks perfectly
                    )
                    
                    # Dynamically fetch the text translations so the colorbar matches the data perfectly
                    tick_texts = [format_label(str(i)) for i in range(1, 6)]
                    
                    fig.update_layout(
                        barmode='stack',
                        barnorm='percent',
                        xaxis_title="Percent (%)",
                        yaxis_title=None,
                        xaxis=dict(),
                        showlegend=False, # Disable discrete legend
                        coloraxis_colorbar=dict(
                            title='',
                            tickmode='array',
                            tickvals=[1, 2, 3, 4, 5],
                            ticktext=tick_texts,
                            ticks='', 
                            thickness=30 # Make it visually blocky like the requested UI
                        )
                    )
                else:
                    fig = px.bar(
                        counts, 
                        y='group', 
                        x='count', 
                        color='response', 
                        orientation='h', 
                        text_auto='.1f',
                        category_orders={"response": ordered_labels}, 
                        color_discrete_sequence=color_palette 
                    )
                    fig.update_layout(
                        barmode='stack',
                        barnorm='percent',
                        xaxis_title="Percent (%)",
                        yaxis_title=None,
                        xaxis=dict(), 
                        showlegend=True,
                        legend_title_text='', 
                        legend=legend_layout 
                    )
            else:
                fig = px.pie(
                    counts, 
                    values='count', 
                    names='response', 
                    hole=0.6,
                    category_orders={"response": ordered_labels}, 
                    color_discrete_sequence=color_palette 
                )
                fig.update_traces(sort=False) 
                fig.update_layout(
                    showlegend=True,
                    legend_title_text='', 
                    legend=legend_layout 
                )

        fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))

        viz_charts.append(html.Div([
            html.Div([ # Wrapped the text in a list so we can stack the question and the new label cleanly
                html.Span(f"{display_question}"),
                html.Br(),
                html.Span(f"[{question_type_label}]", style={'font-size': '11px', 'color': '#888', 'font-weight': 'normal'}) # Injected the tiny tag right below the title
            ], style={'text-align': 'center', 'height': '60px', 'overflow-y': 'auto', 'font-size': '13px', 'font-weight': 'bold'}),
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
            # Receive both categorical and multi-select lists from the dictionary builder
            question_map, option_map, categorical_fields, multi_select_fields = build_survey_dictionaries(tab)
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
                # Store the total time for the entire function
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

        # pass categorical_fields to filter out text inputs like ZIP codes
        viz_charts = populate_survey_charts(df, question_map, option_map, categorical_fields, multi_select_fields, chart_type) 

    # Map the raw key to the clean survey name for the accordion title
    survey_label_map = {
        "UserProfileSurvey": "dfc-onboarding",
        "TripConfirmSurvey": "dfc-gas-trip", 
        "DfcGasTrip": "dfc-gas-trip", 
        "DfcEvRoamingTrip": "dfc-ev-roaming-trip", 
        "DfcEvReturnTrip": "dfc-ev-return-trip" 
    }
    display_name = survey_label_map.get(tab, tab) # Grab the clean name

    return html.Div([
        dmc.Accordion(children=[
            dmc.AccordionItem([
                dmc.AccordionControl(f"Survey Summary Dashboard: {display_name}"), # Replaced raw tab key with the clean display name
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
            uuids_df = pd.DataFrame(store_uuids.get('data', [])) # Added safe get method to prevent KeyError
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
            
            import json # Bring in the JSON library for strict data scrubbing
            clean_records = json.loads(df.to_json(orient='records', date_format='iso')) # forcefully strips out non-serializable Pandas ob            
            result = html.Div([
              dag.AgGrid(
                id={'type': 'data_table', 'id': table_id},
                rowData=clean_records, # feed the perfectly scrubbed records directly to the grid
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