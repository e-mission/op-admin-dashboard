import base64

import dash_bootstrap_components as dbc
import flask
import requests
import dash
import os

from utils import decode_jwt


def get_tokens(code):
    client_id = os.getenv("COGNITO_CLIENT_ID", '')
    client_secret = os.getenv("COGNITO_CLIENT_SECRET", '')
    redirect_uri = os.getenv("COGNITO_REDIRECT_URL", '')
    token_endpoint = os.getenv("COGNITO_TOKEN_ENDPOINT", '')

    encoded_data = base64.b64encode(f'{client_id}:{client_secret}'.encode('ascii')).decode('ascii')
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Basic {encoded_data}'
    }

    body = {
        'grant_type': 'authorization_code',
        'client_id': client_id,
        'code': code,
        'redirect_uri': redirect_uri
    }

    data = {}
    response = requests.post(token_endpoint, data=body, headers=headers)
    try:
        response = response.json()
        id_token = response['id_token']
        user_data = decode_jwt.lambda_handler(id_token)
        data = {
            'id_token': id_token,
            'email': user_data['email'],
        }
    except KeyError:
        pass

    return data


def get_query_params(url_params):
    params = dict()
    if url_params:
        for key ,val in [item.split('=') for item in url_params[1:].split('&')]:
            params[key] = val
    return params


def get_cognito_login_page(text='Welcome to the dashboard', color='black'):
    return [
        dbc.Row([
            dbc.Col([
                dash.html.Label(text, style={
                    'font-size': '15px', 'display': 'block', 'verticalAlign': 'top', 'padding': '15px', 'color': color
                }),
                dbc.Button('Login with AWS Cognito', id='login-button', href=os.getenv("COGNITO_AUTH_URL", ''), style={
                    'font-size': '14px', 'display': 'block', 'padding': '15px', 'verticalAlign': 'top',
                    'background-color': 'green', 'color': 'white'
                }),
            ], style={'display': 'flex', 'justify_content': 'center', 'align-items': 'center',
                      'flex-direction': 'column'}),
        ])
    ]


def authenticate_user(params):
    all_cookies = dict(flask.request.cookies)
    if all_cookies.get('token') is not None:
        user_data = decode_jwt.lambda_handler(all_cookies['token'])
        if user_data:
            return True

    # If code is in query params, validate the user and set the token in cookies
    query_params = get_query_params(params)
    if 'code' in query_params:
        user_data = get_tokens(query_params['code'])
        if user_data.get('id_token') is not None:
            dash.callback_context.response.set_cookie(
                'token',
                user_data['id_token'],
                max_age=60*60,
                httponly=True,
                secure=True,
            )
            return True

    return False
