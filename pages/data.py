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
import plotly.express as px # For donut chart
import urllib.request # For fetching data from urls
import xml.dom.minidom as minidom # For reading xml files
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
        dcc.Store(id='selected-tab', data='tab-users-datatable'),
        dcc.Store(id='loaded-uuids-stats', data=[]),
        dcc.Store(id='all-uuids-stats-loaded', data=False),
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
                    value='analysis/recreated_location',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                ),
            ],
            style={'display': 'none'}
        ),
    ]
)


def clean_location_data(df):
    with ect.Timer() as total_timer:
        if 'data.start_loc.coordinates' in df.columns:
            with ect.Timer() as stage1_timer:
                df['data.start_loc.coordinates'] = df['data.start_loc.coordinates'].apply(lambda x: f'({x[0]}, {x[1]})' if isinstance(x, list) else x)
            esdsq.store_dashboard_time("admin/data/clean_location_data/clean_start_loc_coordinates", stage1_timer)

        if 'data.end_loc.coordinates' in df.columns:
            with ect.Timer() as stage2_timer:
                df['data.end_loc.coordinates'] = df['data.end_loc.coordinates'].apply(lambda x: f'({x[0]}, {x[1]})' if isinstance(x, list) else x)
            esdsq.store_dashboard_time("admin/data/clean_location_data/clean_end_loc_coordinates", stage2_timer)

    esdsq.store_dashboard_time("admin/db_utils/clean_location_data/total_time", total_timer)
    return df

def update_store_trajectories(start_date, end_date, tz, excluded_uuids, key_list):
    with ect.Timer() as total_timer:
        with ect.Timer() as stage1_timer:
            df = query_trajectories(start_date, end_date, tz, key_list)
        esdsq.store_dashboard_time("admin/data/update_store_trajectories/query_trajectories", stage1_timer)

        with ect.Timer() as stage2_timer:
            records = df_to_filtered_records(df, 'user_id', excluded_uuids["data"])
        esdsq.store_dashboard_time("admin/data/update_store_trajectories/filter_records", stage2_timer)

        with ect.Timer() as stage3_timer:
            store = {"data": records, "length": len(records)}
        esdsq.store_dashboard_time("admin/data/update_trajectories/prepare_store_data", stage3_timer)

    esdsq.store_dashboard_time("admin/data/update_store_trajectories/total_time", total_timer)
    return store


@callback(
    Output('keylist-switch-container', 'style'),
    Input('tabs-datatable', 'value'),
)
def show_keylist_switch(tab):
    if tab == 'tab-trajectories-datatable':
        return {'display': 'block'} 
    return {'display': 'none'}


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
    Input('keylist-switch', 'value'),
)
def render_content(tab, store_uuids, store_excluded_uuids, store_trips, store_surveys, store_trajectories, start_date, end_date, timezone, key_list):
    with ect.Timer() as total_timer:
        selected_tab = tab
        logging.debug(f"Callback - {selected_tab} Stage 1: Selected tab updated.")
        content = None

        if tab == 'tab-users-datatable':
            with ect.Timer() as handle_uuids_timer:
                columns = perm_utils.get_uuids_columns()
                users_df = pd.DataFrame(store_uuids['data'])
                if users_df.empty or not perm_utils.has_permission('data_uuids'):
                    content = html.Div([html.P("No data available.")])
                else:
                    users_df = users_df[[c for c in columns if c in users_df.columns]]
                    for col in users_df.columns:
                        if col.endswith('_ts'):
                            users_df[col] = users_df[col].apply(ts_to_iso)
                    if 'total_trips' in users_df.columns and 'labeled_trips' in users_df.columns:
                        loc = users_df.columns.get_loc('labeled_trips') + 1
                        pct = (users_df['labeled_trips'] / users_df['total_trips'])
                        users_df.insert(loc, 'labeled_trips_pct', pct.apply(lambda x: f"{x:.1%}"))
                    content = html.Div([
                        populate_datatable(users_df, store_uuids, 'uuids'),
                        html.P(f"Showing {len(store_uuids['data'])} UUIDs.", style={'margin': '15px 5px'})
                    ])
            esdsq.store_dashboard_time("admin/data/render_content/handle_uuids_tab", handle_uuids_timer)

        elif tab == 'tab-trips-datatable':
            with ect.Timer() as handle_trips_timer:
                data = store_trips.get("data", [])
                columns = perm_utils.get_allowed_trip_columns()
                has_perm = perm_utils.has_permission('data_trips')
                df = pd.DataFrame(data)
                if df.empty and has_perm:
                    content = html.Div([html.Div("No data available", style={'text-align': 'center', 'margin-bottom': '16px'})], style={'margin-top': '36px'})
                elif not has_perm:
                    content = html.Div([html.P("No data available.")])
                else:
                    df = df.drop(columns=[col for col in df.columns if col not in columns])
                    df = clean_location_data(df)
                    trip_labels_enketo = perm_utils.config.get("survey_info", {}).get("trip-labels") == 'ENKETO'
                    if trip_labels_enketo:
                        def extract_response(x):
                            docs = esj.wrapped_loads(x).get('trip_user_input', {}).get('data', {}).get('jsonDocResponse', {})
                            r = next(iter(docs.values()), {})
                            return {k: v for k, v in r.items() if k not in ['meta', 'attrid', 'start', 'end'] and 'xmlns' not in k}
                        response = df['data.user_input'].apply(extract_response)
                        user_input_cols = pd.json_normalize(response)
                    else:
                        user_input_cols = pd.json_normalize(df['data.user_input'].apply(lambda x: esj.wrapped_loads(x) if x is not None else {}))
                    user_input_cols.columns = [f"data.user_input.{col}" for col in user_input_cols.columns]
                    df = pd.concat([df, user_input_cols], axis=1)
                    trips_table = populate_datatable(df, store_uuids, 'trips')
                    content = html.Div([
                        dmc.Checkbox(label="Include human-friendly units for distance and duration", id="humanize-units", checked=True, style={'margin-bottom': '12px'}),
                        dmc.Checkbox(label="Expand user_input to separate columns", id="expand-user-input", checked=False, style={'margin-bottom': '12px'}),
                        trips_table
                    ])
            esdsq.store_dashboard_time("admin/data/render_content/handle_trips_tab", handle_trips_timer)

        elif tab == 'tab-surveys-datatable':
            with ect.Timer() as handle_surveys_timer:
                data = store_surveys.get("data", {})
                if len(data) >= 1:
                    if not perm_utils.has_permission('data_demographics'):
                        content = skeleton(100)
                    else:
                        content = html.Div([
                            dcc.Tabs(id='subtabs-surveys', value=list(data.keys())[0], children=[
                                dcc.Tab(label=key, value=key) for key in data
                            ]),
                            html.Div(id='subtabs-surveys-content')
                        ])
                else:
                    content = None
            esdsq.store_dashboard_time("admin/data/render_content/handle_surveys_tab", handle_surveys_timer)

        elif tab == 'tab-trajectories-datatable':
            with ect.Timer() as handle_trajectories_timer:
                (start_date, end_date) = iso_to_date_only(start_date, end_date)
                if store_trajectories == {}:
                    store_trajectories = update_store_trajectories(start_date, end_date, timezone, store_excluded_uuids, key_list)
                data = store_trajectories["data"]
                if data:
                    columns = list(data[0].keys())
                    df = pd.DataFrame(data)
                    if not df.empty and perm_utils.has_permission('data_trajectories'):
                        df = df.drop(columns=[col for col in df.columns if col not in columns])
                        content = populate_datatable(df, store_uuids, 'trajectories')
                else:
                    content = html.Div([html.Div("No data available", style={'text-align': 'center', 'margin-bottom': '16px'})], style={'margin-top': '36px'})
            esdsq.store_dashboard_time("admin/data/render_content/handle_trajectories_tab", handle_trajectories_timer)

    esdsq.store_dashboard_time("admin/data/render_content/total_time", total_timer)
    return content

# This is the main callback for the surveys tab. It triggers whenever a user 
# clicks a sub-tab and fetches the survey data to build the visual charts and table.
@callback(
    Output('subtabs-surveys-content', 'children'),
    Input('subtabs-surveys', 'value'),
    Input('store-surveys', 'data'),
    Input('store-uuids', 'data')
)
def update_sub_tab(tab, store_surveys, store_uuids):
    # This helper function grabs all the text inside a specific XML node.
    def get_all_text(node):
        parts = []
        for child in node.childNodes:
            if child.nodeType == node.TEXT_NODE:
                parts.append(child.data)
            elif child.nodeType == node.ELEMENT_NODE:
                parts.append(get_all_text(child))
        return "".join(parts).strip()

    # This engine pulls the survey XML from GitHub and maps technical database 
    # IDs to human-readable English questions.
    def build_survey_dictionaries(survey_name):
        try:
            # Access the dynamic configuration for the current study
            config = perm_utils.config 
            form_path = config.get('survey_info', {}).get('surveys', {}).get(survey_name, {}).get('formPath')
            print(f"--- DEBUG: ATTEMPTING TO PARSE XML FROM: {form_path} ---", flush=True)
            
            if not form_path:
                return {}, {}

            # Fetch the raw XML file from the provided URL
            result = urllib.request.urlopen(form_path)
            doc = minidom.parse(result) 
            
            # This part builds a dictionary to translate technical XML IDs into English text.
            itext_map = {}
            for text_node in doc.getElementsByTagName("text"):
                text_id = text_node.getAttribute("id")
                v_nodes = text_node.getElementsByTagName("value")
                if v_nodes and v_nodes[0].firstChild:
                    itext_map[text_id] = v_nodes[0].firstChild.data

            opt_dict, quest_dict = {}, {}
            # Loop through XML tags to match database keys with their corresponding questions
            for tag in ['input', 'select', 'select1']:
                for node in doc.getElementsByTagName(tag):
                    ref = node.getAttribute("ref")
                    if ref:
                        parts = ref.split('/')
                        short_id = parts[-1]
                        full_db_id = ".".join(parts[2:])
                        label_nodes = node.getElementsByTagName("label")
                        if label_nodes:
                            l_ref = label_nodes[0].getAttribute("ref")
                            # Remove itext wrapper syntax to isolate the translation ID
                            clean_id = l_ref.replace("jr:itext('", "").replace("')", "")
                            question_text = itext_map.get(clean_id, short_id)
                            # Map both short and full IDs to ensure the database data is caught
                            quest_dict[short_id] = question_text
                            quest_dict[full_db_id] = question_text
            return quest_dict, opt_dict
        except Exception as e:
            print(f"--- DEBUG ERROR: {e} ---", flush=True)
            return {}, {}

    with ect.Timer() as total_timer:
        surveys_data = store_surveys["data"]
        if tab not in surveys_data or not surveys_data[tab]: return None
        data = surveys_data[tab]
        df = pd.DataFrame(data)
        
        # Build the translation dictionaries using the XML parsing logic
        quest_map, opt_map = build_survey_dictionaries(tab)
        
        # This filter removes any columns that are not defined in the current PSU XML. 
        # This fixes the issue where old demographic questions were being displayed.
        allowed_cols = [c for c in df.columns if c in quest_map or c in ['_id', 'user_id', 'user_token', 'ts']]
        df = df[allowed_cols]
        
        # Build the standard data table view
        table_result = populate_datatable(df, store_uuids, 'surveys')

    # This section processes the survey columns to generate donut charts.
    viz_charts = []
    survey_cols = [c for c in df.columns if c not in ['_id', 'user_id', 'user_token', 'ts']]
    
    for col in survey_cols:
        # Fetch the human-readable question to use as the chart header
        display_question = quest_map.get(col, col.replace('_', ' '))
        if df[col].nunique() < 15:
            # Count the occurrences of each response for the visualization
            counts = df[col].value_counts().reset_index()
            counts.columns = ['response', 'count']
            counts['response'] = counts['response'].apply(lambda x: opt_map.get(str(x), x))
            
            # Build the interactive donut chart using Plotly Express
            fig = px.pie(counts, values='count', names='response', hole=0.6)
            fig.update_traces(textinfo='percent', textfont_size=11)
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=250)
            
            # Create the UI container for the chart including the legend toggle icon
            viz_charts.append(html.Div([
                dmc.ActionIcon(html.I(className="fa fa-chevron-right"), id={'type': 'individual-toggle', 'index': col}, variant="transparent"),
                html.Div(f"Results: {display_question}", 
                         style={'text-align': 'center', 'height': '100px', 'overflow-y': 'auto', 'font-size': '13px', 
                                'padding': '5px', 'margin-bottom': '10px', 'background-color': '#fcfcfc', 'border-bottom': '1px solid #eee'}),
                dcc.Graph(id={'type': 'survey-donut', 'index': col}, figure=fig, config={'displayModeBar': False})
            ], style={'width': '31%', 'display': 'inline-block', 'padding': '15px', 'vertical-align': 'top', 
                      'border': '1px solid #eee', 'border-radius': '8px', 'margin': '1%', 'background-color': '#fff'}))

    # Return the final layout organized inside a collapsible accordion
    return html.Div([
        dmc.Accordion(value="summary-panel", children=[
            dmc.AccordionItem([
                dmc.AccordionControl(f"Survey Summary Dashboard: {tab}"),
                dmc.AccordionPanel([
                    html.Div([
                        dmc.Button("Show All Legends", id="show-legends-btn", variant="outline", size="xs", style={'margin-right': '10px'}),
                        dmc.Button("Hide All Legends", id="hide-legends-btn", variant="outline", size="xs")
                    ], style={'margin-bottom': '20px'}),
                    html.Div(viz_charts, id='survey-charts-container', style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'})
                ])
            ], value="summary-panel")
        ]),
        html.Hr(style={'margin-top': '40px'}),
        html.H4("Detailed Response Table", style={'margin-left': '15px'}),
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
        esdsq.store_dashboard_time("admin/data/populate_datatable/check_dataframe_type", stage1_timer)
        if 'user_token' not in df.columns:
            uuids_df = pd.DataFrame(store_uuids['data'])
            user_id_col = 'data.user_id' if 'data.user_id' in df.columns else 'user_id'
            if user_id_col in df.columns:
                user_id_token_map = users_id_token_map = uuids_df.set_index('user_id')['user_token'].to_dict()
                df.insert(df.columns.get_loc(user_id_col), 'user_token', df[user_id_col].map(user_id_token_map))
        # Stage 2: Create the DataTable from the DataFrame
        with ect.Timer() as stage2_timer:
            df.columns = [col.replace('.', ':') for col in df.columns]
            result = html.Div([
              dag.AgGrid(
                id={'type': 'data_table', 'id': table_id},
                rowData=df.to_dict('records'),
                columnDefs=[{"field": i, "headerName": i.replace('data:', '')} for i in df.columns],
                defaultColDef={ "sortable": True, "filter": True },
                columnSize="autoSize",
                dashGridOptions={"pagination": True, "paginationPageSize": 50, "enableCellTextSelection": True},
                style={"--ag-font-family": "monospace", "height": "600px"},
              ),
              dmc.Button("Download as CSV", id={"type": "download-csv-btn", "id": table_id}, variant='outline', style={'margin-block': '8px'}),
            ])
        esdsq.store_dashboard_time("admin/data/populate_datatable/create_datatable", stage2_timer)
        
    esdsq.store_dashboard_time("admin/data/populate_datatable/total_time", total_timer)
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

# Handles interactivity for survey charts, callback allows users to toggle 
# the legend (color key) for donut charts either globally (all at once) 
# or individually for each specific survey question

@callback(
    Output({'type': 'survey-donut', 'index': MATCH}, 'figure'),
    Input('show-legends-btn', 'n_clicks'),
    Input('hide-legends-btn', 'n_clicks'),
    Input({'type': 'individual-toggle', 'index': MATCH}, 'n_clicks'),
    State({'type': 'survey-donut', 'index': MATCH}, 'figure'),
    prevent_initial_call=True
)
def toggle_legends(show_all, hide_all, individual_click, fig):
    from dash import callback_context
    if not callback_context.triggered: raise PreventUpdate
    trigger_id = callback_context.triggered[0]['prop_id']
    if 'individual-toggle' in trigger_id:
        fig['layout']['showlegend'] = not fig['layout'].get('showlegend', False)
    elif 'show-legends-btn' in trigger_id:
        fig['layout']['showlegend'] = True
    elif 'hide-legends-btn' in trigger_id:
        fig['layout']['showlegend'] = False
    return fig