REQUIRED_NAMED_COLS = [
    {'label': 'trip_start_time_str', 'path': 'data.start_fmt_time'},
    {'label': 'trip_end_time_str', 'path': 'data.end_fmt_time'},
    {'label': 'start_coordinates', 'path': 'data.start_loc.coordinates'},
    {'label': 'end_coordinates', 'path': 'data.end_loc.coordinates'},
]

MULTILABEL_NAMED_COLS = [
    {'label': 'mode_confirm', 'path': 'data.user_input.mode_confirm'},
    {'label': 'purpose_confirm', 'path': 'data.user_input.purpose_confirm'},
    {'label': 'replaced_mode', 'path': 'data.user_input.replaced_mode'},
]

VALID_TRIP_COLS = [
    "data.start_local_dt",
    "data.start_fmt_time",
    "data.end_local_dt",
    "data.end_fmt_time",
    "data.duration",
    "data.distance",
    "data.start_loc.coordinates",
    "data.end_loc.coordinates",
    "user_id"
]

BINARY_TRIP_COLS = [
    'user_id',
    'data.start_place',
    'data.end_place',
]

valid_uuids_columns = [
    'user_token',
    'user_id',
    'update_ts',
    'total_trips',
    'labeled_trips',
    'first_trip',
    'last_trip',
    'last_call',
    'platform',
    'manufacturer',
    'app_version',
    'os_version',
    'phone_lang',
]

BINARY_DEMOGRAPHICS_COLS = [
    'user_id',
    '_id',
]

EXCLUDED_DEMOGRAPHICS_COLS = [
    'data.xmlResponse', 
    'data.name',
    'data.version',
    'data.label',
    'xmlns:jr',
    'xmlns:orx',
    'id',
    'start',
    'end',
    'attrxmlns:jr',
    'attrxmlns:orx',
    'attrid',
    '__version__',
    'attrversion',
    'instanceID',
]