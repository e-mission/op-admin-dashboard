import os
import pandas as pd
import logging

import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback, State, register_page, no_update
import dash_ag_grid as dag

import emission.storage.decorations.token_queries as esdt
import emission.core.get_database as edb
import emcommon.auth.opcode as emcao

from utils.generate_qr_codes import make_qrcode_base64_img, make_qrcodes_zipfile
from utils.permissions import has_permission, config, get_token_prefix


STUDY_CONFIG = os.getenv('STUDY_CONFIG')

if has_permission('token_generate'):
    register_page(__name__, path="/tokens")

token_prefix = get_token_prefix()
configured_subgroups = config.get('opcode', {}).get('subgroups')

layout = html.Div(
    [
        dcc.Store(id="store-tokens", data=[]),
        dcc.Store(id="store-qrcodes", data={}),
        html.Div([
            dcc.Markdown('## Tokens', style={'margin-right': 'auto'}),
            dbc.Button(children='Generate more tokens', id='open-modal-btn', n_clicks=0),
            dbc.Button(children='Export QR codes',
                       color='primary',
                       outline=True,
                       id='token-export', n_clicks=0),
            dcc.Download(id='download-tokens'),
        ],
            style={'display': 'flex', 'gap': '5px', 'margin-bottom': '20px'}
        ),
        html.Div(id='token-table'),
        dbc.Modal([
            dbc.ModalHeader("Generate tokens"),
            dbc.ModalBody([
                html.Div([
                    dbc.Label('Program'),
                    dbc.Input(value=(STUDY_CONFIG or 'program'),
                              id='token-program',
                              type='text',
                              disabled=(bool(STUDY_CONFIG)),
                              required=True),
                ]),
                html.Div([
                    dbc.Label('Token Length'),
                    dbc.Input(value=10, id='token-length', type='number', min=6, max=100, required=True),
                ]),
                html.Div([
                    dbc.Label('Token Subgroup'),
                    dcc.Dropdown(
                        configured_subgroups or ['test'],
                        id='token-subgroup',
                    ),
                ]),
                html.Div([
                    dbc.Label('Number of Tokens'),
                    dbc.Input(value=1, id='token-count', type='number', min=0, required=True),
                ]),
                dbc.Alert(
                    id='generate-tokens-alert',
                    style={'margin': '15px 5px'},
                ),
                html.Div([
                    dbc.Button('Close', id='close-modal-btn', color='primary', outline=True, n_clicks=0),
                    dbc.Button('Generate', id='generate-tokens-btn', color='primary', n_clicks=0),
                ],
                    style={'display': 'flex', 'gap': '10px', 'margin-top': '20px'}
                ),
            ],
                style={'display': 'flex', 'flex-direction': 'column', 'gap': '10px'}
            )
        ],
            id="generate-tokens-modal",
            is_open=False,
        )
    ]
)


@callback(
    Output('store-tokens', 'data'),
    Input('store-tokens', 'data'),
)
def load_tokens(_):
  return [e.get('token') for e in edb.get_token_db().find({})]


@callback(
    Output("generate-tokens-modal", "is_open"),
    [Input("open-modal-btn", "n_clicks"),
     Input("close-modal-btn", "n_clicks"),
     Input("generate-tokens-btn", "n_clicks")],
    [State("generate-tokens-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal(_n1, _n2, _n3, is_open):
    if _n1 or _n2 or _n3: return not is_open


@callback(
    Output('generate-tokens-btn', 'disabled'),
    Output('generate-tokens-alert', 'children'),
    Output('generate-tokens-alert', 'color'),
    Input('token-program', 'value'),
    Input('token-subgroup', 'value'),
    Input('token-length', 'value'),
    Input('token-count', 'value'),
)
def validate_token_inputs(program, subgroup, token_length, token_count):
    if not program:
        return True, f'Program must be {STUDY_CONFIG or "program"}', 'danger'
    elif configured_subgroups and (not subgroup or subgroup not in configured_subgroups):
        return True,f'Subgroup must be one of {configured_subgroups}', 'danger'
    elif not token_length or token_length < 6 or token_length > 100:
        return True, 'Token length must be between 6 and 100', 'danger'
    elif not token_count or token_count < 1:
        return True, 'Token count must be at least 1', 'danger'
    
    example_token = emcao.generate_opcode(token_prefix, program, subgroup, token_length)
    info = [
        dbc.Label(
            f'{token_count} token(s) will be generated in this format:',
            style={'margin': '0'}
        ),
        html.Hr(style={'margin': '8px'}),
        dbc.Label(
            example_token,
            style={'font-family': 'monospace',
                   'word-break': 'break-all',
                   'margin': '0' }
        ),
    ]
    return False, info, 'info'


@callback(
    Output('store-tokens', 'data', allow_duplicate=True),
    Output('generate-tokens-btn', 'n_clicks'),
    Input('generate-tokens-btn', 'n_clicks'),
    Input('token-program', 'value'),
    Input('token-subgroup', 'value'),
    Input('token-length', 'value'),
    Input('token-count', 'value'),
    State('store-tokens', 'data'),
    Input('store-uuids', 'data'),
    prevent_initial_call=True
)
def generate_tokens(n_clicks, program, subgroup, token_length, token_count, tokens, uuids):
    if n_clicks is not None and n_clicks > 0:
        new_tokens = [
            emcao.generate_opcode(token_prefix, program, subgroup, token_length)
            for _ in range(token_count)
        ]
        esdt.insert_many_tokens(new_tokens)
        tokens += new_tokens
        return tokens, 0
    return no_update, no_update


@callback(
    Output('download-tokens', 'data'),
    Input('token-export', 'n_clicks'),
    State('store-tokens', 'data'),
    prevent_initial_call=True,
)
def export_tokens(_, tokens):
    zip_fn = make_qrcodes_zipfile(tokens)
    return dcc.send_bytes(zip_fn, "tokens.zip")


@callback(
    Output('token-table', 'children'),
    Input('store-uuids', 'data'),
    Input('store-tokens', 'data'),
    Input('store-qrcodes', 'data'),
)
def populate_datatable(uuids, tokens, qrcodes):
    logging.info(f'tokens: {tokens}')
    if not tokens:
        return None
    df = pd.DataFrame({'token': tokens})
    df['qr_code'] = df['token'].map(qrcodes).fillna('(click to reveal)')
    uuids_records = uuids.get('data', [])
    df['in_use'] = df.apply(
        lambda row: any(uuid['user_email'] == row['token']
                        for uuid in uuids_records),
        axis=1,
    )
    df = df.reindex(columns=['token', 'in_use', 'qr_code'])
    return html.Div([
        dag.AgGrid(
            id='tokens-table',
            rowData=df.to_dict('records'),
            columnDefs=[{"field": c, "headerName": c,
                         "minWidth": 500 if c == 'qr_code' else 0} for c in df.columns],
            defaultColDef={"sortable": True, "filter": True,
                          "cellRenderer": "markdown",
                          "autoHeight": True},
            dashGridOptions={"pagination": True, "enableCellTextSelection": True},
            columnSize="autoSize",
            style={"height": "700px",
                   "--ag-font-family": "monospace"},
            getRowId="params.data.token",
        ),
        dbc.Button("Download CSV", id="export-tokens-table-btn",
                   color="primary", outline=True, className="mt-3"),
    ])


@callback(
    Output("tokens-table", "exportDataAsCsv"),
    Output("tokens-table", "csvExportParams"),
    Input("export-tokens-table-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_table_as_csv(_):
    return True, {"fileName": "tokens-table.csv"}


@callback(
    Output("store-qrcodes", "data"),
    Input("tokens-table", "cellClicked"),
    State("store-qrcodes", "data"),
)
def make_qr_on_row_selected(cell_clicked, qrcodes):
    if not cell_clicked or cell_clicked["colId"] != "qr_code":
        return no_update
    token = cell_clicked["rowId"]
    if token in qrcodes:
        return no_update
    img_src = f'data:image/png;base64,{make_qrcode_base64_img(token)}'
    qrcodes[token] = f'![{token}]({img_src})'
    return qrcodes
