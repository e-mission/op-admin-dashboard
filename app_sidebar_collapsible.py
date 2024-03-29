"""
This app creates an animated sidebar using the dbc.Nav component and some local CSS. Each menu item has an icon, when
the sidebar is collapsed the labels disappear and only the icons remain. Visit www.fontawesome.com to find alternative
icons to suit your needs!

dcc.Location is used to track the current location, a callback uses the current location to render the appropriate page
content. The active prop of each NavLink is set automatically according to the current pathname. To use this feature you
must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash documentation: https://dash.plot.ly/urls
"""
import os
import arrow

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, Dash
import dash_auth
import logging
# Set the logging right at the top to make sure that debug
# logs are displayed in dev mode
# until https://github.com/plotly/dash/issues/532 is fixed
if os.getenv('DASH_DEBUG_MODE', 'True').lower() == 'true':
    logging.basicConfig(level=logging.DEBUG)

from utils.datetime_utils import iso_to_date_only
from utils.db_utils import df_to_filtered_records, query_uuids, query_confirmed_trips, query_demographics
from utils.permissions import has_permission
import flask_talisman as flt



OPENPATH_LOGO = "https://www.nrel.gov/transportation/assets/images/openpath-logo.jpg"
auth_type = os.getenv('AUTH_TYPE')


if auth_type == 'cognito':
    from utils.cognito_utils import authenticate_user, get_cognito_login_page
elif auth_type == 'basic':
    from config import VALID_USERNAME_PASSWORD_PAIRS

app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    use_pages=True,
)
server = app.server  # expose server variable for Procfile

if auth_type == 'basic':
    auth = dash_auth.BasicAuth(
        app,
        VALID_USERNAME_PASSWORD_PAIRS
    )


sidebar = html.Div(
    [
        html.Div(
            [
                # width: 3rem ensures the logo is the exact width of the
                # collapsed sidebar (accounting for padding)
                html.Img(src=OPENPATH_LOGO, style={"width": "3rem"}),
                html.H2("OpenPATH"),
            ],
            className="sidebar-header",
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.I(className="fas fa-home me-2"), 
                        html.Span("Overview")
                    ],
                    href=dash.get_relative_path("/"),
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-sharp fa-solid fa-database me-2"),
                        html.Span("Data"),
                    ],
                    href=dash.get_relative_path("/data"),
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-right-to-bracket me-2"),
                        html.Span("Tokens"),
                    ],
                    href=dash.get_relative_path("/tokens"),
                    active="exact",
                    style={'display': 'block' if has_permission('token_generate') else 'none'},
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-globe me-2"),
                        html.Span("Map"),
                    ],
                    href=dash.get_relative_path("/map"),
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-hourglass me-2"),
                        html.Span("Segment trip time"),
                    ],
                    href=dash.get_relative_path("/segment_trip_time"),
                    active="exact",
                    style={'display': 'block' if has_permission('segment_trip_time') else 'none'},
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-envelope-open-text me-2"),
                        html.Span("Push notification"),
                    ],
                    href=dash.get_relative_path("/push_notification"),
                    active="exact",
                    style={'display': 'block' if has_permission('push_send') else 'none'},
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-gear me-2"),
                        html.Span("Settings"),
                    ],
                    href=dash.get_relative_path("/settings"),
                    active="exact",
                )
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

# Global controls including date picker and timezone selector
def make_controls():
  # according to docs, DatePickerRange will accept YYYY-MM-DD format
  today_date = arrow.now().format('YYYY-MM-DD')
  last_week_date = arrow.now().shift(days=-7).format('YYYY-MM-DD')
  tomorrow_date = arrow.now().shift(days=1).format('YYYY-MM-DD')
  return html.Div([
      html.Div([
          # Global Date Picker
          dcc.DatePickerRange(
              id='date-picker',
              display_format='D MMM Y',
              start_date=last_week_date,
              end_date=today_date,
              min_date_allowed='2010-1-1',
              max_date_allowed=tomorrow_date,
              initial_visible_month=today_date,
          ),
          dbc.Button(
              html.I(className="fas fa-bars", id='collapse-icon'),
              outline=True,
              id="collapse-button",
              n_clicks=0,
              style={'color': '#444', 'border': '1px solid #dbdbdb',
                    'border-radius': '3px', 'margin-left': '3px'}
          ),
      ],
          style={'display': 'flex'},
      ),
      dbc.Collapse([
          html.Div([
              html.Span('Query trips using: ', style={'margin-right': '10px'}),
              dcc.Dropdown(
                  id='date-picker-timezone',
                  options=[
                      {'label': 'UTC Time', 'value': 'utc'},
                      {'label': 'My Local Timezone', 'value': 'local'},
                      # {'label': 'Local Timezone of Trips', 'value': 'trips'},
                  ],
                  value='utc',
                  clearable=False,
                  searchable=False,
                  style={'width': '180px'},
              )]
          ),

          dcc.Checklist(
              id='global-filters',
              options=[
                  {'label': 'Exclude "test" users',
                   'value': 'exclude-test-users'},
              ],
              value=['exclude-test-users'],
              style={'margin-top': '10px'},
          ),
      ],
          id='collapse-filters',
          is_open=False,
          style={'padding': '5px 15px 10px', 'border': '1px solid #dbdbdb', 'border-top': '0'}
      ),
  ],
      style={'margin': '10px 10px 0 auto',
             'width': 'fit-content',
             'display': 'flex',
             'flex-direction': 'column'}
  )

page_content = dcc.Loading(
    type='default',
    fullscreen=True,
    children=html.Div(dash.page_container, style={
        "margin-left": "5rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    })
)


def make_home_page(): return [
    sidebar,
    html.Div([make_controls(), page_content])
]


def make_layout(): return html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='store-trips', data={}),
    dcc.Store(id='store-uuids', data={}),
    dcc.Store(id='store-excluded-uuids', data={}), # if 'test' users are excluded, a list of their uuids
    dcc.Store(id='store-demographics', data={}),
    dcc.Store(id='store-trajectories', data={}),
    html.Div(id='page-content', children=make_home_page()),
])
app.layout = make_layout

# make the 'filters' menu collapsible
@app.callback(
    Output("collapse-filters", "is_open"),
    Output("collapse-icon", "className"),
    [Input("collapse-button", "n_clicks")],
    [Input("collapse-filters", "is_open")],
)
def toggle_collapse_filters(n, is_open):
    if not n: return (is_open, "fas fa-bars")
    if is_open:
      return (False, "fas fa-bars")
    else:
      return (True, "fas fa-chevron-up")

# Load data stores
@app.callback(
    Output("store-uuids", "data"),
    Output("store-excluded-uuids", "data"),
    Input('date-picker', 'start_date'),  # these are ISO strings
    Input('date-picker', 'end_date'),  # these are ISO strings
    Input('date-picker-timezone', 'value'),
    Input('global-filters', 'value'),
)
def update_store_uuids(start_date, end_date, timezone, filters):
    (start_date, end_date) = iso_to_date_only(start_date, end_date)
    dff = query_uuids(start_date, end_date, timezone)
    if dff.empty:
        return {"data": [], "length": 0}, {"data": [], "length": 0}
    # if 'exclude-testusers' filter is active,
    # exclude any rows with user_token containing 'test', and
    # output a list of those excluded UUIDs so other callbacks can exclude them too
    if 'exclude-test-users' in filters:
        excluded_uuids_list = dff[dff['user_token'].str.contains(
            'test')]['user_id'].tolist()
    else:
        excluded_uuids_list = []
    records = df_to_filtered_records(dff, 'user_id', excluded_uuids_list)
    store_uuids = {
        "data": records,
        "length": len(records),
    }
    store_excluded_uuids = {
        "data": excluded_uuids_list,
        "length": len(excluded_uuids_list),
    }
    return store_uuids, store_excluded_uuids


@app.callback(
    Output("store-demographics", "data"),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('date-picker-timezone', 'value'),
    Input('store-excluded-uuids', 'data'),
)
def update_store_demographics(start_date, end_date, timezone, excluded_uuids):
    dataframes = query_demographics()
    records = {}
    for key, df in dataframes.items():
        records[key] = df_to_filtered_records(df, 'user_id', excluded_uuids["data"])
    store = {
        "data": records,
        "length": len(records),
    }
    return store


# Note: this triggers twice on load, not great with a slow db
@app.callback(
    Output("store-trips", "data"),
    Input('date-picker', 'start_date'), # these are ISO strings
    Input('date-picker', 'end_date'), # these are ISO strings
    Input('date-picker-timezone', 'value'),
    Input('store-excluded-uuids', 'data'),
)
def update_store_trips(start_date, end_date, timezone, excluded_uuids):
    (start_date, end_date) = iso_to_date_only(start_date, end_date)
    df = query_confirmed_trips(start_date, end_date, timezone)
    records = df_to_filtered_records(df, 'user_id', excluded_uuids["data"])
    # logging.debug("returning records %s" % records[0:2])
    store = {
        "data": records,
        "length": len(records),
    }
    return store


# Define the callback to display the page content based on the URL path
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'search'),
)
def display_page(search):
    if auth_type == 'cognito':
        try:
            is_authenticated = authenticate_user(search)
        except Exception as e:
            print(e)
            return get_cognito_login_page('Unsuccessful authentication, try again.', 'red')

        if is_authenticated:
            return make_home_page()
        return get_cognito_login_page()

    return make_home_page()

extra_csp_url = [
    "https://raw.githubusercontent.com",
    "https://*.tile.openstreetmap.org",
    "https://cdn.jsdelivr.net",
    "https://use.fontawesome.com",
    "https://www.nrel.gov",
    "data:",
    "blob:"
]
csp = {
       'default-src': ["'self'", "'unsafe-inline'"] + extra_csp_url
      }

flt.Talisman(server, content_security_policy=csp, strict_transport_security=False)

if __name__ == "__main__":
    envPort = int(os.getenv('DASH_SERVER_PORT', '8050'))
    envDebug = os.getenv('DASH_DEBUG_MODE', 'True').lower() == 'true'
    app.logger.setLevel(logging.DEBUG)
    logging.debug("before override, current server config = %s" % server.config)
    server.config.update(
        TESTING=envDebug,
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True
    )
    logging.debug("after override, current server config = %s" % server.config)
    app.run_server(debug=envDebug, host='0.0.0.0', port=envPort)
