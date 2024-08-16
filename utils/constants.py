REQUIRED_NAMED_COLS = [
    {'label': 'trip_start_time_str', 'path': 'data.start_fmt_time'},
    {'label': 'trip_end_time_str', 'path': 'data.end_fmt_time'},
    {'label': 'start_coordinates', 'path': 'data.start_loc.coordinates'},
    {'label': 'end_coordinates', 'path': 'data.end_loc.coordinates'},
]

MULTILABEL_NAMED_COLS = [
]

VALID_TRIP_COLS = [
    "data.start_local_dt",
    "data.start_fmt_time",
    "data.end_local_dt",
    "data.end_fmt_time",
    "data.duration",
    "data.duration_seconds",
    "data.distance",
    "data.distance_km",
    "data.distance_miles",
    "data.distance_meters",
    "data.start_loc.coordinates",
    "data.end_loc.coordinates",
    "data.primary_sensed_mode",
    "data.primary_predicted_mode",
    "data.primary_ble_sensed_mode",
    "user_id"
]

BINARY_TRIP_COLS = [
    'user_id',
    'data.start_place',
    'data.end_place',
    "cleaned_section_summary",
    "inferred_section_summary",
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

EXCLUDED_TRAJECTORIES_COLS = [
    'data.loc.type',
    'data.loc.coordinates',
    'data.local_dt.year',
    'data.local_dt.month',
    'data.local_dt.day',
    'data.local_dt.hour',
    'data.local_dt.minute',
    'data.local_dt.second',
    'data.local_dt.weekday',
    'data.local_dt.timezone',
    'data.local_dt_year',
    'data.local_dt_month',
    'data.local_dt_day',
    'data.local_dt_hour',
    'data.local_dt_minute',
    'data.local_dt_second',
    'data.local_dt_weekday',
    'data.local_dt_timezone',
]
