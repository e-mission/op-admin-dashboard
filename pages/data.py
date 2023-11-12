"""
Note that the callback will trigger even if prevent_initial_call=True. This is because dcc.Location must be in app.py.
Since the dcc.Location component is not in the layout when navigating to this page, it triggers the callback.
The workaround is to check if the input value is None.
"""
from dash import dcc, html, Input, Output, callback, register_page, dash_table
from datetime import date
# Etc
import logging
import pandas as pd
from dash.exceptions import PreventUpdate

from utils import permissions as perm_utils
from utils import db_utils
from utils.db_utils import query_trajectories
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
    ]
)


def clean_location_data(df):
    if 'data.start_loc.coordinates' in df.columns:
        df['data.start_loc.coordinates'] = df['data.start_loc.coordinates'].apply(lambda x: f'({x[0]}, {x[1]})')
    if 'data.end_loc.coordinates' in df.columns:
        df['data.end_loc.coordinates'] = df['data.end_loc.coordinates'].apply(lambda x: f'({x[0]}, {x[1]})')
    return df

def update_store_trajectories(start_date_obj,end_date_obj):
    global store_trajectories
    df = query_trajectories(start_date_obj,end_date_obj)
    records = df.to_dict("records")   
    store = {
        "data": records,
        "length": len(records),
    }
    store_trajectories = store
    return store


@callback(
    Output('tabs-content', 'children'),
    Input('tabs-datatable', 'value'),
    Input('store-uuids', 'data'),
    Input('store-trips', 'data'),
    Input('store-demographics', 'data'),
    Input('store-trajectories', 'data'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),

)
def render_content(tab, store_uuids, store_trips, store_demographics, store_trajectories, start_date, end_date):
    data, columns, has_perm = None, [], False
    if tab == 'tab-uuids-datatable':
        data = store_uuids["data"]
        data = db_utils.add_user_stats(data)
        columns = perm_utils.get_uuids_columns()
        has_perm = perm_utils.has_permission('data_uuids')
    elif tab == 'tab-trips-datatable':
        data = store_trips["data"]
        columns = perm_utils.get_allowed_trip_columns()
        columns.update(
            col['label'] for col in perm_utils.get_allowed_named_trip_columns()
        )
        has_perm = perm_utils.has_permission('data_trips')
    elif tab == 'tab-demographics-datatable':
        data = store_demographics["data"]
        # if only one survey is available, process it without creating a subtab
        if len(data) == 1:   
            data = data[0]
            columns = list(data[0].keys())
            has_perm = perm_utils.has_permission('data_demographics')
        # for multiple survey, create subtabs for unique surveys
        else:
            return html.Div([
                dcc.Tabs(id='subtabs-demographics', value=next(iter(data)), children=[
                    dcc.Tab(label= key, value= key) for key in data
            ]),  
                html.Div(id='subtabs-demographics-content')
            ]) 
    elif tab == 'tab-trajectories-datatable':
        # Currently store_trajectories data is loaded only when the respective tab is selected
        #Here we query for trajectory data once "Trajectories" tab is selected
        start_date_obj = date.fromisoformat(start_date) if start_date else None
        end_date_obj = date.fromisoformat(end_date) if end_date else None
        if store_trajectories == {}:
            store_trajectories = update_store_trajectories(start_date_obj,end_date_obj)
        data = store_trajectories["data"]
        if data:
            columns = list(data[0].keys())
            columns = perm_utils.get_trajectories_columns(columns)
            has_perm = perm_utils.has_permission('data_trajectories')
       
    df = pd.DataFrame(data)
    if df.empty or not has_perm:
        return None

    df = df.drop(columns=[col for col in df.columns if col not in columns])
    df = clean_location_data(df)

    return populate_datatable(df)

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
            has_perm = perm_utils.has_permission('data_demographics')

        df = pd.DataFrame(data)
        if df.empty or not has_perm:
            return None

        df = df.drop(columns=[col for col in df.columns if col not in columns])

        return populate_datatable(df)
      
def populate_datatable(df):
    if not isinstance(df, pd.DataFrame):
        raise PreventUpdate
    return dash_table.DataTable(
        # id='my-table',
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
        style_table={'overflowX': 'auto'}
    )
