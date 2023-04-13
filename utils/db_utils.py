from datetime import datetime, timezone

import pandas as pd

import emission.core.get_database as edb

from utils.perm_utils import get_trips_columns, get_additional_trip_columns


def query_uuids(start_date, end_date):
    query = {'update_ts': {'$exists': True}}
    if start_date is not None:
        start_time = datetime.combine(start_date, datetime.min.time()).astimezone(timezone.utc)
        query['update_ts']['$gte'] = start_time

    if end_date is not None:
        end_time = datetime.combine(end_date, datetime.max.time()).astimezone(timezone.utc)
        query['update_ts']['$lt'] = end_time

    projection = {
        '_id': 0,
        'user_id': '$uuid',
        'user_token': '$user_email',
        'update_ts': 1
    }

    query_result = edb.get_uuid_db().find(query, projection)
    df = pd.json_normalize(list(query_result))
    if not df.empty:
        df['update_ts'] = pd.to_datetime(df['update_ts'])
        df['user_id'] = df['user_id'].apply(str)
    return df

def query_confirmed_trips(start_date, end_date):
    query = {
        '$and': [
            {'metadata.key': 'analysis/confirmed_trip'},
            {'data.start_ts': {'$exists': True}},
        ]
    }
    if start_date is not None:
        start_time = datetime.combine(start_date, datetime.min.time())
        query['$and'][1]['data.start_ts']['$gte'] = start_time.timestamp()

    if end_date is not None:
        end_time = datetime.combine(end_date, datetime.max.time())
        query['$and'][1]['data.start_ts']['$lt'] = end_time.timestamp()

    projection = {
        '_id': 0,
        'user_id': 1,
        'trip_start_time_str': '$data.start_fmt_time',
        'trip_end_time_str': '$data.end_fmt_time',
        'timezone': '$data.start_local_dt.timezone',
        'start_coordinates': '$data.start_loc.coordinates',
        'end_coordinates': '$data.end_loc.coordinates',
    }

    for column in get_trips_columns():
        projection[column] = 1

    for column in get_additional_trip_columns():
        projection[column['label']] = column['path']

    query_result = edb.get_analysis_timeseries_db().find(query, projection)
    df = pd.json_normalize(list(query_result))
    if not df.empty:
        df['user_id'] = df['user_id'].apply(str)
        if 'data.start_place' in df.columns:
            df['data.start_place'] = df['data.start_place'].apply(str)
        if 'data.end_place' in df.columns:
            df['data.end_place'] = df['data.end_place'].apply(str)
    return df
