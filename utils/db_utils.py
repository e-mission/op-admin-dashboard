import logging
import arrow
from uuid import UUID

import pandas as pd
import pymongo

import emission.core.get_database as edb
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.timeseries.timequery as estt
import emission.core.wrapper.motionactivity as ecwm


from utils import constants
from utils import permissions as perm_utils
from utils.datetime_utils import iso_range_to_ts_range

def query_uuids(start_date: str, end_date: str, tz: str):
    # As of now, time filtering does not apply to UUIDs; we just query all of them.
    # Vestigial code commented out and left below for future reference

    # logging.debug("Querying the UUID DB for %s -> %s" % (start_date,end_date))
    # query = {'update_ts': {'$exists': True}}
    # if start_date is not None:
    #     # have arrow create a datetime using start_date and time 00:00:00 in UTC
    #     start_time = arrow.get(start_date).datetime
    #     query['update_ts']['$gte'] = start_time
    # if end_date is not None:
    #     # have arrow create a datetime using end_date and time 23:59:59 in UTC
    #     end_time = arrow.get(end_date).replace(hour=23, minute=59, second=59).datetime
    #     query['update_ts']['$lt'] = end_time
    # projection = {
    #     '_id': 0,
    #     'user_id': '$uuid',
    #     'user_token': '$user_email',
    #     'update_ts': 1
    # }

    logging.debug("Querying the UUID DB for (no date range)")

    # This should actually use the profile DB instead of (or in addition to)
    # the UUID DB so that we can see the app version, os, manufacturer...
    # I will write a couple of functions to get all the users in a time range
    # (although we should define what that time range should be) and to merge
    # that with the profile data
    entries = edb.get_uuid_db().find()
    df = pd.json_normalize(list(entries))
    if not df.empty:
        df['update_ts'] = pd.to_datetime(df['update_ts'])
        df['user_id'] = df['uuid'].apply(str)
        df['user_token'] = df['user_email']
        df.drop(columns=["uuid", "_id"], inplace=True)
    return df

def query_confirmed_trips(start_date: str, end_date: str, tz: str):
    (start_ts, end_ts) = iso_range_to_ts_range(start_date, end_date, tz)
    ts = esta.TimeSeries.get_aggregate_time_series()
    # Note to self, allow end_ts to also be null in the timequery
    # we can then remove the start_time, end_time logic
    entries = ts.find_entries(
        key_list=["analysis/confirmed_trip"],
        time_query=estt.TimeQuery("data.start_ts", start_ts, end_ts),
    )
    df = pd.json_normalize(list(entries))

    # logging.debug("Before filtering, df columns are %s" % df.columns)
    if not df.empty:
        columns = [col for col in perm_utils.get_all_trip_columns() if col in df.columns]
        df = df[columns]
        # logging.debug("After getting all columns, they are %s" % df.columns)
        for col in constants.BINARY_TRIP_COLS:
            if col in df.columns:
                df[col] = df[col].apply(str)
        for named_col in perm_utils.get_all_named_trip_columns():
            if named_col['path'] in df.columns:
                df[named_col['label']] = df[named_col['path']]
                # df = df.drop(columns=[named_col['path']])
        # TODO: We should really display both the humanized value and the raw value
        # humanized value for people to see the entries in real time
        # raw value to support analyses on the downloaded data
        # I still don't fully grok which columns are displayed
        # https://github.com/e-mission/op-admin-dashboard/issues/29#issuecomment-1530105040
        # https://github.com/e-mission/op-admin-dashboard/issues/29#issuecomment-1530439811
        # so just replacing the distance and duration with the humanized values for now
        df['data.distance_meters'] = df['data.distance']
        use_imperial = perm_utils.config.get("display_config",
            {"use_imperial": False}).get("use_imperial", False)
        # convert to km to humanize
        df['data.distance_km'] = df['data.distance'] / 1000
        # convert km further to miles because this is the US, Liberia or Myanmar
        # https://en.wikipedia.org/wiki/Mile
        df['data.duration_seconds'] = df['data.duration']
        if use_imperial:
            df['data.distance_miles'] = df['data.distance_km'] * 0.6213712

        df['data.duration'] = df['data.duration'].apply(lambda d: arrow.utcnow().shift(seconds=d).humanize(only_distance=True))

    # logging.debug("After filtering, df columns are %s" % df.columns)
    # logging.debug("After filtering, the actual data is %s" % df.head())
    # logging.debug("After filtering, the actual data is %s" % df.head().trip_start_time_str)
    return df

def query_demographics():
    # Returns dictionary of df where key represent differnt survey id and values are df for each survey
    logging.debug("Querying the demographics for (no date range)")
    ts = esta.TimeSeries.get_aggregate_time_series()

    entries = ts.find_entries(["manual/demographic_survey"])
    data = list(entries)

    available_key = {}
    for entry in data:
        survey_key = list(entry['data']['jsonDocResponse'].keys())[0]
        if survey_key not in available_key:
            available_key[survey_key] = []
        available_key[survey_key].append(entry)

    dataframes = {}
    for key, json_object in available_key.items():
        df = pd.json_normalize(json_object)
        dataframes[key] = df

    for key, df in dataframes.items():
        if not df.empty:
            for col in constants.BINARY_DEMOGRAPHICS_COLS:
                if col in df.columns:
                    df[col] = df[col].apply(str) 
            columns_to_drop = [col for col in df.columns if col.startswith("metadata")]
            df.drop(columns= columns_to_drop, inplace=True) 
            modified_columns = perm_utils.get_demographic_columns(df.columns)  
            df.columns = modified_columns 
            df.columns=[col.rsplit('.',1)[-1] if col.startswith('data.jsonDocResponse.') else col for col in df.columns]  
            for col in constants.EXCLUDED_DEMOGRAPHICS_COLS:
                if col in df.columns:
                    df.drop(columns= [col], inplace=True) 
                    
    return dataframes

def query_trajectories(start_date: str, end_date: str, tz: str):
    
    (start_ts, end_ts) = iso_range_to_ts_range(start_date, end_date, tz)
    ts = esta.TimeSeries.get_aggregate_time_series()
    entries = ts.find_entries(
        key_list=["analysis/recreated_location"],
        time_query=estt.TimeQuery("data.ts", start_ts, end_ts),
    )
    df = pd.json_normalize(list(entries))
    if not df.empty:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(str)
        columns_to_drop = [col for col in df.columns if col.startswith("metadata")]
        df.drop(columns= columns_to_drop, inplace=True) 
        for col in constants.EXCLUDED_TRAJECTORIES_COLS:
            if col in df.columns:
                df.drop(columns= [col], inplace=True) 
        df['data.mode_str'] = df['data.mode'].apply(lambda x: ecwm.MotionTypes(x).name if x in set(enum.value for enum in ecwm.MotionTypes) else 'UNKNOWN')
    return df


def add_user_stats(user_data):
    for user in user_data:
        user_uuid = UUID(user['user_id'])

        total_trips = esta.TimeSeries.get_aggregate_time_series().find_entries_count(
            key_list=["analysis/confirmed_trip"],
            extra_query_list=[{'user_id': user_uuid}]
        )
        user['total_trips'] = total_trips

        labeled_trips = esta.TimeSeries.get_aggregate_time_series().find_entries_count(
            key_list=["analysis/confirmed_trip"],
            extra_query_list=[{'user_id': user_uuid}, {'data.user_input': {'$ne': {}}}]
        )
        user['labeled_trips'] = labeled_trips

        profile_data = edb.get_profile_db().find_one({'user_id': user_uuid})
        user['platform'] = profile_data.get('curr_platform')
        user['manufacturer'] = profile_data.get('manufacturer')
        user['app_version'] = profile_data.get('client_app_version')
        user['os_version'] = profile_data.get('client_os_version')
        user['phone_lang'] = profile_data.get('phone_lang')




        if total_trips > 0:
            time_format = 'YYYY-MM-DD HH:mm:ss'
            ts = esta.TimeSeries.get_time_series(user_uuid)
            start_ts = ts.get_first_value_for_field(
                key='analysis/confirmed_trip',
                field='data.end_ts',
                sort_order=pymongo.ASCENDING
            )
            if start_ts != -1:
                user['first_trip'] = arrow.get(start_ts).format(time_format)

            end_ts = ts.get_first_value_for_field(
                key='analysis/confirmed_trip',
                field='data.end_ts',
                sort_order=pymongo.DESCENDING
            )
            if end_ts != -1:
                user['last_trip'] = arrow.get(end_ts).format(time_format)

            last_call = ts.get_first_value_for_field(
                key='stats/server_api_time',
                field='data.ts',
                sort_order=pymongo.DESCENDING
            )
            if last_call != -1:
                user['last_call'] = arrow.get(last_call).format(time_format)

    return user_data
