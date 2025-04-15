"""
Note that the callback will trigger even if prevent_initial_call=True. This is because dcc.Location must
be in app.py.  Since the dcc.Location component is not in the layout when navigating to this page, it triggers the callback.
The workaround is to check if the input value is None.

"""

import os
import re
import dash
from dash import dcc, html, Input, Output, State, callback, register_page
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

import emission.analysis.configs.dynamic_config as eacd

from utils.config_update_utils import trigger_config_update_workflow, get_workflow_run_status, get_pr_status, get_recent_workflow_run, get_recent_pr


register_page(__name__, path="/settings")


layout = html.Div(
    [
        dcc.Markdown('## Settings'),
        dbc.Button("Manage Admin Access",
                   id="admin-access-button", color="primary"),
        dbc.Modal([
            dbc.ModalHeader("Manage Admin Access"),
            dbc.ModalBody([
                dcc.Store(id='config-update-status', data=None),
                dcc.Interval(id='config-update-interval',
                             interval=5000, n_intervals=0),
                dcc.Markdown('Admin users:'),
                dag.AgGrid(
                    id='admin-grid',
                    rowData=[],
                    columnDefs=[{'field': 'email'}],
                    style={'height': '300px'},
                ),
                dbc.Form([
                    dbc.Row(
                        [
                            dbc.Label("Add Email", width="auto"),
                            dbc.Col(
                                dbc.Input(type="email", id="email",
                                          placeholder="Enter email"),
                                className="me-3",
                            ),
                            dbc.FormText(
                                'An invitation will be sent to the email address with ' +
                                'a temporary password and instructions to access the admin dashboard. ' +
                                'Only invite trusted users.',
                                color="secondary",
                            ),
                            dbc.Col([
                                dbc.Button(
                                    "Submit", id="admin-access-submit-button", color="primary"),
                                dbc.Button("Close", id="admin-access-close-button", color="primary", outline=True),],
                                width="auto",
                                className="d-flex gap-2",
                            ),
                        ],
                        className="g-2 my-2 d-flex gap-2",
                    ),
                    dbc.Row(id='alert-row', className="px-2",
                            style={'white-space': 'break-spaces'}),
                ]),
            ]),
        ],
            id="admin-access-modal",
        ),
    ]
)


@callback(
    Output("admin-access-modal", "is_open"),
    Input("admin-access-button", "n_clicks"),
    Input("admin-access-close-button", "n_clicks"),
    State("admin-access-modal", "is_open"),
)
def toggle_modal(_n1, _n2, is_open):
    if _n1 or _n2:
        return not is_open


@callback(
    Output('admin-grid', 'rowData'),
    Input('admin-grid', 'id'),
    Input('config-update-status', 'data'),
)
def get_current_admins(_, data):
    if data and not data.get('merged_at'):
        return dash.no_update
    eacd.dynamic_config = None
    config = eacd.get_dynamic_config()
    emails = config.get('admin_dashboard', {}).get('admin_access', [])
    return [{'email': e} for e in emails]


@callback(
    Output("admin-access-submit-button", "disabled"),
    Input("email", "value"),
)
def disable_submit_button(email):
    if email and re.match(r'^[\w\.-]+@[\w\.-]+\.\w{2,}$', email):
        return False
    return True


@callback(
    Output("config-update-status", "data"),
    Output("email", "value"),
    Input("admin-access-submit-button", "n_clicks"),
    State("email", "value"),
    prevent_initial_call=True,
)
def submit_email(_, email):
    status_code = trigger_config_update_workflow({
        'deployment': os.getenv('STUDY_CONFIG'),
        'script_name': 'update_admin_access',
        'script_args': f'add {email}',
    })
    if status_code == 204:
        return {'id': None}, None
    elif status_code == 500:
        return {'error': 'Failed to authenticate'}, None
    else:
        return {'error': 'Unknown error'}, None


@callback(
    Output("config-update-interval", "disabled"),
    Output("alert-row", "children"),
    Input("config-update-status", "data"),
)
def show_status(data):
    if not data:
        return True, None
    if 'error' in data:
        return True, dbc.Alert(data['error'], color="danger")
    url = data.get('html_url')
    link = html.A(url, href=url, className="alert-link", target="_blank")
    if data.get('merged_at'):
        return True, dbc.Alert(["Admin access updated:\n", link, "\nIt may take a few minutes for updates to appear in the dashboard."], color="success")
    if data.get('state') == 'closed' or data.get('conclusion') == 'failure':
        return True, dbc.Alert(["Admin access update failed:\n", link], color="danger")
    if 'timed_out' in data:
        return True, dbc.Alert(["Polling admin access update timed out. Please check status manually: \n", link], color="warning")
    return False, dbc.Alert([dbc.Spinner(), " Updating admin access...\n", link], color="info")


@callback(
    Output("config-update-status", "data", allow_duplicate=True),
    Input("config-update-interval", "n_intervals"),
    State("config-update-status", "data"),
    prevent_initial_call=True,
)
def poll_status(n_intervals, data):
    if not data or n_intervals > 15:
        return None
    if not data['id']:
        return get_recent_workflow_run() or dash.no_update
    if 'state' in data and data['state'] != 'closed':
        return get_pr_status(data['number']) or dash.no_update
    if 'conclusion' in data and data['conclusion'] is None:
        return get_workflow_run_status(data['id'])
    if 'conclusion' in data and data['conclusion'] == 'success':
        return get_recent_pr(data['id']) or dash.no_update
    return dash.no_update
