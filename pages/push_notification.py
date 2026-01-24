"""
Note that the callback will trigger even if prevent_initial_call=True. This is because dcc.Location must
be in app.py.  Since the dcc.Location component is not in the layout when navigating to this page, it triggers the callback.
The workaround is to check if the input value is None.

"""
from uuid import UUID

from dash import dcc, html, Input, Output, State, callback, register_page, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import json

import emission.storage.decorations.user_queries as esdu
import emission.core.wrapper.user as ecwu
import emission.net.ext_service.push.notify_usage as pnu
from utils.permissions import has_permission, config

configured_subgroups = config.get('opcode', {}).get('subgroups')

(uuids_perm, tokens_perm) = has_permission('options_uuids'), has_permission('options_emails')
if has_permission('push_send'):
    register_page(__name__, path="/push_notification")

intro = """
## Push notification
"""

layout = html.Div([
    dcc.Markdown(intro),
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dbc.Label('Filter Subgroup', style={'padding-top': '5px'}),
                    dcc.Dropdown(multi=True, id='push-subgroup-filter',
                                options=configured_subgroups or ['test'],
                                style={'display': 'block' if tokens_perm else 'none'}),
                ], style={'flex': '1'}),
                html.Div([
                    dbc.Label('Filter Specific Users', style={'padding-top': '5px'}),
                    dcc.Dropdown(multi=True, id='push-users-filter'),
                ], style={'flex': '1'}),
            ], style={'display': 'flex', 'gap': '10px'}),
        ]),

        html.Div([
            dbc.Label('Push Notification Actions'),
            dbc.Checklist(
                options={
                    "notify": "Visible Notification",
                    "config-update": "Trigger Config Update",
                    # "prompt-survey": "Prompt Survey",
                },
                value=['notify'],
                id='push-actions',
            ),
        ]),

        html.Div([
            dbc.Label('Title'),
            dbc.Input(id='push-title',
                      placeholder='Enter push notification title',
                      className='mb-2'),
            dbc.Label('Message'),
            dbc.Textarea(id='push-message',
                         placeholder='Enter push notification message'),
        ],
            id='notify-options',
            style={'display': 'none'},
        ),

        html.Div([
            dbc.Label('Minimum allowed config version'),
            dbc.Input(id='min-config-version',
                      type='number',
                      value=config['version'],
                      min=1,
                      max=config['version']),
            dbc.FormText(f'The latest config version is {config["version"]}'),
        ],
            id='config-update-options',
            style={'display': 'none'},
        ),

        # html.Div([
        #     dbc.Label('Prompt Survey Options'),
        # ],
        #     id='prompt-survey-options',
        #     style={'display': 'none'},
        # ),
        dbc.Alert(
            [html.Span(id='push-info'),
             html.Pre(id='push-payload-json', className='mb-0')],
            id='push-info-alert',
            color="info",
            is_open=True,
        ),

        html.Div([
            dbc.Checklist(
                id='push-log-options',
                options={'dry-run': 'Dry Run (print logs but do not send)'},
                value=[],
                className='mb-2',
            ),
            dbc.Button(
                'Send',
                id='push-send-button',
                color='primary',
                n_clicks=0,
            ),
        ]),

        dbc.Alert(
            html.Pre(id='push-log'),
            id='push-log-alert',
            is_open=False,
            color='secondary',
        ),
    ],
        style={
            'display': 'flex',
            'flex-direction': 'column',
            'gap': '15px',
            'padding': '20px',
        },
    ),
])


@callback(
    Output('push-users-filter', 'options'),
    Input('store-uuids', 'data'),
)
def populate_users(uuids_data):
    uuids_df = pd.DataFrame(uuids_data.get('data'))
    if uuids_perm and tokens_perm:
        labels = (uuids_df['user_token'] + '\n(' + uuids_df['user_id'] + ')').tolist()
    elif tokens_perm:
        labels = uuids_df['user_token'].tolist()
    else:
        labels = uuids_df['user_id'].tolist()
    return {val: label for val, label in zip(uuids_df['user_id'], labels)}


@callback(
    Output('push-users-filter', 'value'),
    Input('push-subgroup-filter', 'value'),
    Input('push-users-filter', 'options'),
)
def filter_users_by_subgroup(selected_subgroups, user_options):
    if not selected_subgroups:
        return []
    return [user for user, label in user_options.items()
            if any(f'_{subgroup}_' in label for subgroup in selected_subgroups)]


def get_push_payload(push_actions, title, message, min_config_version):
    """
    All keys AND values must be strings or firebase will reject the payload.
    """
    actions = {}
    if 'notify' in push_actions and title and message:
        actions['title'] = title
        actions['message'] = message
    if 'config-update' in push_actions and min_config_version:
        actions['min_config_version'] = str(min_config_version)
    return actions if actions else None


@callback(
    Output('push-info', 'children'),
    Output('push-info-alert', 'color'),
    Output('push-payload-json', 'children'),
    Output('push-send-button', 'disabled'),
    Input('push-users-filter', 'value'),
    Input('push-users-filter', 'options'),
    Input('push-actions', 'value'),
    Input('push-title', 'value'),
    Input('push-message', 'value'),
    Input('min-config-version', 'value'),
)
def update_push_info(selected_users, users_options, push_actions, title, message, min_config_version):
    text = ""
    if selected_users:
        text += f"{len(selected_users)} users"
    else:
        text += f"All {len(users_options)} users"

    payload = get_push_payload(push_actions, title, message, min_config_version)
    if payload:
        text += f" will receive a push notification with:"
    else:
        text += " selected – no actions specified"
        return text, 'warning', payload, True

    return text, 'info', json.dumps(payload, indent=2), False


@callback(
    Output('notify-options', 'style'),
    Output('config-update-options', 'style'),
    # Output('prompt-survey-options', 'style'),
    Input('push-actions', 'value'),
)
def toggle_push_options(selected_push_actions):
    return [
        {'display': 'block' if a in selected_push_actions else 'none'}
        for a in ['notify', 'config-update']  # , 'prompt-survey']
    ]


@callback(
    Output('push-log', 'children'),
    Output('push-log-alert', 'is_open'),
    Output('push-send-button', 'n_clicks'),
    Input('push-send-button', 'n_clicks'),
    State('push-users-filter', 'value'),
    State('push-log-options', 'value'),
    State('push-actions', 'value'),
    State('push-title', 'value'),
    State('push-message', 'value'),
    State('min-config-version', 'value'),
)
def send_push_notification(send_n_clicks, user_uuids, log_options, push_actions, title, message, min_config_version):
    if send_n_clicks > 0:
        logs = [f'Push Title: {title}', f'Push Message: {message}', f'Push Options: {push_actions}']

        uuid_list = [UUID(uuid_str) for uuid_str in user_uuids] if user_uuids else esdu.get_all_uuids()
        if uuids_perm:
            uuid_str_list = [str(uuid_val) for uuid_val in uuid_list]
            logs.append(f"About to send push to uuid list = {uuid_str_list}")
        if tokens_perm:
            token_list = [ecwu.User.fromUUID(uuid_val)._User__email for uuid_val in uuid_list if uuid_val is not None]
            logs.append(f"About to send push to token list = {token_list}")

        payload = get_push_payload(push_actions, title, message, min_config_version)
        if payload:
            logs.append(f'Payload: {json.dumps(payload, indent=2)}')

        if 'dry-run' in log_options:
            logs.append("dry run, skipping actual push")
            return "\n".join(logs), True, 0
        else:
            print(payload)
            response = pnu.send_visible_notification_to_users(
                uuid_list,
                title,
                message,
                payload,
            )
            pnu.display_response(response)
            logs.append("Push notification sent successfully")
            return "\n".join(logs), True, 0
    return no_update, no_update, no_update
