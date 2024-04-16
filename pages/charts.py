from dash import dcc, html, Input, Output, callback, register_page, Dash
import pandas as pd
import pygwalker as pyg

from dash.dash import no_update
from utils import permissions as perm_utils


register_page(__name__, path="/charts")

intro = """## Charts"""


layout = html.Div([
   html.H1('PygWalker Charts'),
   dcc.Dropdown(
      id='data-options',
      options=[
         {'label': 'store_trips', 'value': 'store_trips'},
         {'label': 'store_user_stats', 'value': 'store_user_stats'}
      ],
      value = 'store_trips'
   ),
   html.Iframe(id='charts', srcDoc = '', style={'width':'100vw', 'height':'100vh'})
])

@callback(
   Output('charts', 'srcDoc'),
   Input('store-user-stats', 'data'),
   Input('store-trips', 'data'),
   Input('data-options', 'value')
)
def generate_charts(store_user_stats, store_trips, selected_option):
   if selected_option == 'store_trips':
      if not store_trips:
         return no_update
      data = store_trips["data"]
      columns = perm_utils.get_allowed_trip_columns()
      columns.update(
            col['label'] for col in perm_utils.get_allowed_named_trip_columns()
         )
      has_perm = perm_utils.has_permission('data_trips')

   if selected_option == 'store_user_stats':
      if not store_user_stats:
         return no_update
      data = store_user_stats["data"]
      columns = perm_utils.get_uuids_columns()
      has_perm = perm_utils.has_permission('data_uuids')
   
   if not has_perm:
      return no_update
   df = pd.DataFrame(data)
   df = df.drop(columns=[col for col in df.columns if col not in columns])
   walker = pyg.walk(df, return_html=True, hide_Data_Source_Config=False)
   return walker