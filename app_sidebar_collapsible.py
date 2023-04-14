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
from datetime import date

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, Dash
import dash_auth

from utils.db_utils import query_uuids, query_confirmed_trips
from utils.permissions import has_permission



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
                    href="/",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-sharp fa-solid fa-database me-2"),
                        html.Span("Data"),
                    ],
                    href="/data",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-right-to-bracket me-2"),
                        html.Span("Tokens"),
                    ],
                    href="/tokens",
                    active="exact",
                    style={'display': 'block' if has_permission('token_generate') else 'none'},
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-globe me-2"),
                        html.Span("Map"),
                    ],
                    href="/map",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-solid fa-envelope-open-text me-2"),
                        html.Span("Push notification"),
                    ],
                    href="/push_notification",
                    active="exact",
                    style={'display': 'block' if has_permission('push_send') else 'none'},
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-gear me-2"),
                        html.Span("Settings"),
                    ],
                    href="/settings",
                    active="exact",
                )
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)


content = html.Div([
    # Global Date Picker
    html.Div(
        dcc.DatePickerRange(
            id='date-picker',
            display_format='D/M/Y',
            start_date_placeholder_text='D/M/Y',
            end_date_placeholder_text='D/M/Y',
            min_date_allowed=date(2010, 1, 1),
            max_date_allowed=date.today(),
            initial_visible_month=date.today(),
        ), style={'margin': '10px 10px 0 0', 'display': 'flex', 'justify-content': 'right'}
    ),

    # Pages Content
    dcc.Loading(
        type='default',
        fullscreen=True,
        children=html.Div(dash.page_container, style={
            "margin-left": "5rem",
            "margin-right": "2rem",
            "padding": "2rem 1rem",
        })
    ),
])


home_page = [
    sidebar,
    content,
]


app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='store-trips', data={}),
        dcc.Store(id='store-uuids', data={}),
        html.Div(id='page-content', children=home_page),
    ]
)


# Load data stores
@app.callback(
    Output("store-uuids", "data"),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
)
def update_store_uuids(start_date, end_date):
    start_date_obj = date.fromisoformat(start_date) if start_date else None
    end_date_obj = date.fromisoformat(end_date) if end_date else None
    dff = query_uuids(start_date_obj, end_date_obj)
    records = dff.to_dict("records")
    store = {
        "data": records,
        "length": len(records),
    }
    return store


@app.callback(
    Output("store-trips", "data"),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
)
def update_store_trips(start_date, end_date):
    start_date_obj = date.fromisoformat(start_date) if start_date else None
    end_date_obj = date.fromisoformat(end_date) if end_date else None
    df = query_confirmed_trips(start_date_obj, end_date_obj)
    records = df.to_dict("records")
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
            return home_page
        return get_cognito_login_page()

    return home_page



if __name__ == "__main__":
    envPort = int(os.getenv('DASH_SERVER_PORT', '8050'))
    envDebug = os.getenv('DASH_DEBUG_MODE', 'True').lower() == 'true'
    app.run_server(debug=envDebug, host='0.0.0.0', port=envPort)
