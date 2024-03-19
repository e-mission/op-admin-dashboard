"""
Note that the callback will trigger even if prevent_initial_call=True. This is because dcc.Location must
be in app.py.  Since the dcc.Location component is not in the layout when navigating to this page, it triggers the callback.
The workaround is to check if the input value is None.

"""
from uuid import UUID

from dash import dcc, html, Input, Output, State, callback, register_page
import pandas as pd

import emission.storage.decorations.user_queries as esdu
import emission.core.wrapper.user as ecwu
import emission.net.ext_service.push.notify_usage as pnu
from utils.permissions import has_permission


if has_permission('push_send'):
    register_page(__name__, path="/push_notification")

intro = """
## Push notification
"""

push_receiver_options = [{'label': 'All users', 'value': 'all'}]
if has_permission('options_emails'):
    push_receiver_options.append({'label': 'User Emails', 'value': 'email'})
if has_permission('options_uuids'):
    push_receiver_options.append({'label': 'User UUIDs', 'value': 'uuid'})


layout = html.Div([
    dcc.Markdown(intro),
    html.Div([
        html.Div(children=[
            html.Label('Sending to:'),
            dcc.RadioItems(
                className='radio-items',
                id='push-receiver-options',
                options=push_receiver_options,
                value='all',
                style={
                    'padding': '5px',
                    'margin': 'auto'
                }
            ),

            html.Div([
                html.Label('User Emails', style={'padding-top': '5px'}),
                dcc.Dropdown(multi=True, disabled=True, id='push-user-emails'),
            ], style={'display': 'block' if has_permission('options_emails') else 'none'}),

            html.Div([
                html.Label('UUIDs', style={'padding-top': '5px'}),
                dcc.Dropdown(multi=True, disabled=True, id='push-user-uuids'),
            ], style={'display': 'block' if has_permission('options_uuids') else 'none'}),

            html.Br(),
            html.Label('Survey Specs'),
            dcc.Dropdown(options=["Notify", "Survey", "Popup", "Website"], value='Notify', id='push-survey-spec'),

            html.Br(),
            dcc.Checklist(
                className='radio-items',
                id='push-log-options',
                options=[
                    {'label': 'Show UUIDs', 'value': 'show-uuids'},
                    {'label': 'Show Emails', 'value': 'show-emails'},
                    {'label': 'Dry Run', 'value': 'dry-run'},
                ],
                value=['show-uuids'],
                style={
                    'padding': '5px',
                    'margin': 'auto'
                }
            ),

            html.Label('Log Messages'),
            dcc.Textarea(value='We can follow sending push here', id='push-log', disabled=True, style={
                'font-size': '14px', 'width': '100%', 'display': 'block', 'margin-bottom': '10px',
                'margin-right': '5px', 'height':'200px', 'verticalAlign': 'top', 'background-color': '#d4c49b',
                'overflow': 'hidden',
            })
        ], style={'padding': 10, 'flex': 1}),

        html.Div(children=[
            html.Label('Title'),
            html.Br(),
            dcc.Textarea(value='', id='push-title', style={
                'font-size': '14px', 'width': '100%', 'display': 'block', 'margin-bottom': '10px',
                'margin-right': '5px', 'height': '30px', 'verticalAlign': 'top', 'background-color': '#b4dbf0',
                'overflow': 'hidden',
            }),

            html.Label('Message'),
            html.Br(),
            dcc.Textarea(value='', id='push-message', style={
                'font-size': '14px', 'width': '100%', 'display': 'block', 'margin-bottom': '10px',
                'margin-right': '5px', 'height':'100px', 'verticalAlign': 'top', 'background-color': '#b4dbf0',
                'overflow': 'hidden',
            }),
            html.Br(),

            html.Button(children='Send', id='push-send-button', n_clicks=0, style={
                'font-size': '14px', 'width': '140px', 'display': 'block', 'margin-bottom': '10px',
                'margin-right': '5px', 'height':'40px', 'verticalAlign': 'top', 'background-color': 'green',
                'color': 'white',
            }),
            html.Button(children='Clear Message', id='push-clear-message-button', n_clicks=0, style={
                'font-size': '14px', 'width': '140px', 'display': 'block', 'margin-bottom': '10px',
                'margin-right': '5px', 'height':'40px', 'verticalAlign': 'top', 'background-color': 'red',
                'color': 'white',
            }),
        ], style={'padding': 10, 'flex': 1})
    ], style={'display': 'flex', 'flex-direction': 'row'})
])

@callback(
    Output('push-user-emails', 'disabled'),
    Output('push-user-uuids', 'disabled'),
    Input('push-receiver-options', 'value'),
)
def handle_receivers(value):
    emails_disabled = True
    uuids_disabled = True
    if value == 'email':
        emails_disabled = False
    elif value == 'uuid':
        uuids_disabled = False
    return emails_disabled, uuids_disabled


@callback(
    Output('push-user-emails', 'options'),
    Output('push-user-uuids', 'options'),
    Input('store-uuids', 'data'),
)
def populate_data(uuids_data):
    emails = list()
    uuids = list()
    uuids_df = pd.DataFrame(uuids_data.get('data'))
    if has_permission('options_emails'):
        emails = uuids_df['user_token'].tolist()
    if has_permission('options_uuids'):
        uuids = uuids_df['user_id'].tolist()
    return emails, uuids


@callback(
    Output('push-message', 'value'),
    Input('push-clear-message-button', 'n_clicks'),
)
def clear_push_message(n_clicks):
    return ''


@callback(
    Output('push-log', 'value'),
    Output('push-send-button', 'n_clicks'),
    Input('push-send-button', 'n_clicks'),
    State('push-log', 'value'),
    State('push-receiver-options', 'value'),
    State('push-user-emails', 'value'),
    State('push-user-uuids', 'value'),
    State('push-log-options', 'value'),
    State('push-title', 'value'),
    State('push-message', 'value'),
    State('push-survey-spec', 'value',)
)
def send_push_notification( send_n_clicks, log, query_spec, emails, uuids, log_options, title, message, survey_spec):
    if send_n_clicks > 0:
        logs = [f'Push Title: {title}', f'Push Message: {message}', f'Survey Spec: {survey_spec}']
        if query_spec == 'all':
            uuid_list = esdu.get_all_uuids()
        elif query_spec == 'email':
            uuid_list = [ecwu.User.fromEmail(email).uuid for email in emails]
        elif query_spec == 'uuid':
            uuid_list = [UUID(uuid_str) for uuid_str in uuids]
        else:
            uuid_list = []

        if 'show-uuids' in log_options:
            uuid_str_list = [str(uuid_val) for uuid_val in uuid_list]
            logs.append(f"About to send push to uuid list = {uuid_str_list}")
        if 'show-emails' in log_options:
            email_list = [ecwu.User.fromUUID(uuid_val)._User__email for uuid_val in uuid_list if uuid_val is not None]
            logs.append(f"About to send push to email list = {email_list}")

        if 'dry-run' in log_options:
            logs.append("dry run, skipping actual push")
            return "\n".join(logs), 0
        else:
            response = pnu.send_visible_notification_to_users(
                uuid_list,
                title,
                message,
                survey_spec,
            )
            pnu.display_response(response)
            logs.append("Push notification sent successfully")
            return "\n".join(logs), 0
    return log, 0
