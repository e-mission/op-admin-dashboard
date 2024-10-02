import logging
import arrow
from uuid import UUID
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import pymongo
import time
import emission.core.get_database as edb
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.timeseries.aggregate_timeseries as estag
import emission.storage.timeseries.timequery as estt
import emission.core.wrapper.motionactivity as ecwm
import emission.storage.timeseries.geoquery as estg
import emission.storage.decorations.section_queries as esds

from utils import constants
from utils import permissions as perm_utils
from utils.datetime_utils import iso_range_to_ts_range
from functools import lru_cache

def df_to_filtered_records(df, col_to_filter=None, vals_to_exclude=None):
    start_time = time.time()  # Start timing for the entire function

    # Check if df is a valid DataFrame and if it is empty
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        logging.debug("Utils - df_to_filtered_records Stage 1: Input is not a valid DataFrame or is empty.")
        return []
    
    # Default to an empty list if vals_to_exclude is None
    if vals_to_exclude is None:
        vals_to_exclude = []

    # Perform filtering if col_to_filter and vals_to_exclude are provided
    if col_to_filter and vals_to_exclude:
        logging.debug(f"Utils - df_to_filtered_records Stage 2: Filtering on column '{col_to_filter}' with values to exclude: {vals_to_exclude}")
        
        # Ensure vals_to_exclude is a list of strings
        if not isinstance(vals_to_exclude, list) or not all(isinstance(val, str) for val in vals_to_exclude):
            raise ValueError("vals_to_exclude must be a list of strings.")
        
        df_before_filtering = df.copy()  # Save a copy for logging purposes
        df = df[~df[col_to_filter].isin(vals_to_exclude)]
        logging.debug(f"Utils - df_to_filtered_records Stage 3: Rows before filtering: {len(df_before_filtering)}, Rows after filtering: {len(df)}")

    # Return the filtered DataFrame as a list of dictionaries
    end_time = time.time()  # End timing
    execution_time = end_time - start_time
    logging.debug(f'Utils - df_to_filtered_records Stage 4: Total time taken for df_to_filtered_records: {execution_time:.4f} seconds')
    
    return df.to_dict("records")


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

    logging.debug("Utils - UUIDs Stage 1: Querying the UUID DB for (no date range)")
    # This should actually use the profile DB instead of (or in addition to)
    # the UUID DB so that we can see the app version, os, manufacturer...
    # I will write a couple of functions to get all the users in a time range
    # (although we should define what that time range should be) and to merge
    # that with the profile data
    start_time = time.time()  # Start timing for the entire function
    entries_start_time = time.time()  # Start timing for fetching entries
    entries = edb.get_uuid_db().find()
    logging.debug(f"Utils - UUIDs Stage 2: Time taken for fetching UUID entries: {time.time() - entries_start_time:.4f} seconds")

    df = pd.json_normalize(list(entries))
    if not df.empty:
        df['update_ts'] = pd.to_datetime(df['update_ts'])
        df['user_id'] = df['uuid'].apply(str)
        df['user_token'] = df['user_email']
        df.drop(columns=["uuid", "_id"], inplace=True)

    end_time = time.time()  # End timing
    execution_time = end_time - start_time
    logging.debug(f'Utils - UUIDs Stage 3: Total time taken for query_uuids: {execution_time:.4f} seconds')

    return df


def query_confirmed_trips(start_date: str, end_date: str, tz: str):
    start_time = time.time()  # Start timer for the entire function
    logging.debug("Utils - Confirmed Trips Stage 1: Converting date range to timestamps")
    
    (start_ts, end_ts) = iso_range_to_ts_range(start_date, end_date, tz)
    ts_start_time = time.time()  # Start timing for time series object retrieval
    ts = esta.TimeSeries.get_aggregate_time_series()
    # Note to self, allow end_ts to also be null in the timequery
    # we can then remove the start_time, end_time logic
    logging.debug(f"Utils - Confirmed Trips Stage 2: Time taken for get_aggregate_time_series: {time.time() - ts_start_time:.4f} seconds")

    entries_start_time = time.time()  # Start timing for retrieving data
    df = ts.get_data_df("analysis/confirmed_trip",
                         time_query=estt.TimeQuery("data.start_ts", start_ts, end_ts))
    logging.debug(f"Utils - Confirmed Trips Stage 3: Time taken for get_data_df: {time.time() - entries_start_time:.4f} seconds")

    user_input_cols = []

    logging.debug("Utils - Confirmed Trips Stage 4: Before filtering, df columns are %s" % df.columns)
    if not df.empty:
        # Since we use `get_data_df` instead of `pd.json_normalize`,
        # we lose the "data" prefix on the fields and they are only flattened one level
        # Here, we restore the prefix for the VALID_TRIP_COLS from constants.py
        # for backwards compatibility. We do this for all columns since columns which don't exist are ignored by the rename command.
        rename_cols = constants.VALID_TRIP_COLS
        rename_mapping = dict(zip([c.replace("data.", "") for c in rename_cols], rename_cols))
        logging.debug("Utils - Confirmed Trips Stage 5: Rename mapping is %s" % rename_mapping)
        df.rename(columns=rename_mapping, inplace=True)
        logging.debug("Utils - Confirmed Trips Stage 6: After renaming columns, they are %s" % df.columns)

        # Now copy over the coordinates
        coords_start_time = time.time()
        df['data.start_loc.coordinates'] = df['start_loc'].apply(lambda g: g["coordinates"])
        df['data.end_loc.coordinates'] = df['end_loc'].apply(lambda g: g["coordinates"])
        logging.debug(f"Utils - Confirmed Trips Stage 7: Time taken for copying coordinates: {time.time() - coords_start_time:.4f} seconds")

        # Add primary modes from the sensed, inferred and ble summaries. Note that we do this
        # **before** filtering the `all_trip_columns` because the
        # *_section_summary columns are not currently valid
        
        # Check if 'md' is not a dictionary or does not contain the key 'distance'
        # or if 'md["distance"]' is not a dictionary.
        # If any of these conditions are true, return "INVALID".
        get_max_mode_from_summary = lambda md: (
            "INVALID"
            if not isinstance(md, dict)
            or "distance" not in md
            or not isinstance(md["distance"], dict)
            else (
                # Get the maximum value from 'md["distance"]' using the values of 'md["distance"].get' as the key for 'max'.
                # This operation only happens if the length of 'md["distance"]' is greater than 0.
                # Otherwise, return "INVALID".
                max(md["distance"], key=md["distance"].get)
                if len(md["distance"]) > 0
                else "INVALID"
            )
        )

        mode_processing_start_time = time.time()
        df["data.primary_sensed_mode"] = df.cleaned_section_summary.apply(get_max_mode_from_summary)
        df["data.primary_predicted_mode"] = df.inferred_section_summary.apply(get_max_mode_from_summary)
        if 'ble_sensed_summary' in df.columns:
            df["data.primary_ble_sensed_mode"] = df.ble_sensed_summary.apply(get_max_mode_from_summary)
        else:
            logging.debug("Utils - Confirmed Trips Stage 8: No BLE support found, ignoring...")

        logging.debug(f"Utils - Confirmed Trips Stage 9: Time taken for mode processing: {time.time() - mode_processing_start_time:.4f} seconds")

        # Expand the user inputs
        user_input_start_time = time.time()
        user_input_df = pd.json_normalize(df.user_input)
        df = pd.concat([df, user_input_df], axis='columns')
        logging.debug(f"Utils - Confirmed Trips Stage 10: Before filtering user input {user_input_df.columns=}")
        user_input_cols = [c for c in user_input_df.columns
            if "metadata" not in c and
               "xmlns" not in c and
               "local_dt" not in c and
               'xmlResponse' not in c and
               "_id" not in c]
        logging.debug(f"Utils - Confirmed Trips Stage 11: After filtering user input {user_input_cols=}")

        combined_col_list = list(perm_utils.get_all_trip_columns()) + user_input_cols
        logging.debug(f"Utils - Confirmed Trips Stage 12: Combined list {combined_col_list=}")
        columns = [col for col in combined_col_list if col in df.columns]
        df = df[columns]
        logging.debug(f"Utils - Confirmed Trips Stage 13: After filtering against the combined list {df.columns=}")

        # Convert binary trip columns to string
        for col in constants.BINARY_TRIP_COLS:
            if col in df.columns:
                df[col] = df[col].apply(str)

        # Add named trip columns
        for named_col in perm_utils.get_all_named_trip_columns():
            if named_col['path'] in df.columns:
                df[named_col['label']] = df[named_col['path']]
        
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

    end_time = time.time()  # End timing
    execution_time = end_time - start_time
    logging.debug(f'Utils - Confirmed Trips Stage 14: Total time taken for query_confirmed_trips: {execution_time:.4f} seconds')

    return (df, user_input_cols)


def query_demographics():
    start_time = time.time()  # Start timer for the entire function
    logging.debug("Utils - Demographics Stage 1: Querying the demographics for (no date range)")

    ts_start_time = time.time()  # Start timing for time series object retrieval
    ts = esta.TimeSeries.get_aggregate_time_series()
    logging.debug(f"Utils - Demographics Stage 2: Time taken for get_aggregate_time_series: {time.time() - ts_start_time:.4f} seconds")

    entries_start_time = time.time()  # Start timing for finding entries
    entries = ts.find_entries(["manual/demographic_survey"])
    data = list(entries)
    logging.debug(f"Utils - Demographics Stage 3: Time taken for find_entries: {time.time() - entries_start_time:.4f} seconds")

    available_key = {}
    for entry in data:
        survey_key = list(entry['data']['jsonDocResponse'].keys())[0]
        if survey_key not in available_key:
            available_key[survey_key] = []
        available_key[survey_key].append(entry)

    dataframes = {}
    for key, json_object in available_key.items():
        df_normalize_start_time = time.time()  # Start timing for normalization
        df = pd.json_normalize(json_object)
        dataframes[key] = df
        logging.debug(f"Utils - Demographics Stage 4: Time taken for json_normalize for key '{key}': {time.time() - df_normalize_start_time:.4f} seconds")

        if not df.empty:
            # Convert object-type columns to strings
            col_convert_start_time = time.time()
            for col in constants.BINARY_DEMOGRAPHICS_COLS:
                if col in df.columns:
                    df[col] = df[col].apply(str)
            logging.debug(f"Utils - Demographics Stage 5: Time taken for converting binary demographic columns to strings: {time.time() - col_convert_start_time:.4f} seconds")

            # Drop metadata columns
            drop_metadata_start_time = time.time()
            columns_to_drop = [col for col in df.columns if col.startswith("metadata")]
            df.drop(columns=columns_to_drop, inplace=True)
            logging.debug(f"Utils - Demographics Stage 6: Time taken for dropping metadata columns: {time.time() - drop_metadata_start_time:.4f} seconds")

            # Rename columns
            rename_start_time = time.time()
            modified_columns = perm_utils.get_demographic_columns(df.columns)
            df.columns = modified_columns
            df.columns = [col.rsplit('.', 1)[-1] if col.startswith('data.jsonDocResponse.') else col for col in df.columns]
            logging.debug(f"Utils - Demographics Stage 7: Time taken for renaming columns: {time.time() - rename_start_time:.4f} seconds")

            # Drop excluded columns
            drop_excluded_start_time = time.time()
            for col in constants.EXCLUDED_DEMOGRAPHICS_COLS:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            logging.debug(f"Utils - Demographics Stage 8: Time taken for dropping excluded columns: {time.time() - drop_excluded_start_time:.4f} seconds")

    total_time = time.time() - start_time  # End timer for the entire function
    logging.debug(f"Utils - Demographics Stage 9: Total time taken for query_demographics: {total_time:.4f} seconds")

    return dataframes


def query_trajectories(start_date: str, end_date: str, tz: str, key_list):
    start_time = time.time()  # Start timer for the entire function

    # Convert the date range to timestamps
    (start_ts, end_ts) = iso_range_to_ts_range(start_date, end_date, tz)
    logging.debug(f"Utils - Trajectory Stage 1: Time taken for iso_range_to_ts_range: {time.time() - start_time:.4f} seconds")

    # Get the time series object
    ts = esta.TimeSeries.get_aggregate_time_series()
    logging.debug(f"Utils - Trajectory Stage 2: Time taken for get_aggregate_time_series: {time.time() - start_time:.4f} seconds")

    # Convert key_list to a list if it's not already
    key_list = [key_list]
    
    # Find entries in the time series data
    entries_start_time = time.time()  # Start timing for this section
    entries = ts.find_entries(
        key_list=key_list,
        time_query=estt.TimeQuery("data.ts", start_ts, end_ts),
    )

    logging.debug(f"Utils - Trajectory Stage 3: Time taken for find_entries: {time.time() - entries_start_time:.4f} seconds")

    # Normalize entries into a DataFrame
    df_normalize_start_time = time.time()
    df = pd.json_normalize(list(entries))
    logging.debug(f"Utils - Trajectory Stage 4: Time taken for json_normalize: {time.time() - df_normalize_start_time:.4f} seconds")

    if not df.empty:
        # Convert object-type columns to strings
        col_convert_start_time = time.time()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(str)
        logging.debug(f"Utils - Trajectory Stage 5: Time taken for converting object columns to strings: {time.time() - col_convert_start_time:.4f} seconds")

        # Drop metadata columns
        drop_metadata_start_time = time.time()
        columns_to_drop = [col for col in df.columns if col.startswith("metadata")]
        df.drop(columns=columns_to_drop, inplace=True)
        logging.debug(f"Utils - Trajectory Stage 6: Time taken for dropping metadata columns: {time.time() - drop_metadata_start_time:.4f} seconds")

        # Drop excluded columns
        drop_excluded_start_time = time.time()
        for col in constants.EXCLUDED_TRAJECTORIES_COLS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        logging.debug(f"Utils - Trajectory Stage 7: Time taken for dropping excluded columns: {time.time() - drop_excluded_start_time:.4f} seconds")

        # Process 'background/location'
        location_check_start_time = time.time()
        if 'background/location' in key_list:
            if 'data.mode' in df.columns:
                df['data.mode'] = ''  # Set to blank
        else:
            df['data.mode_str'] = df['data.mode'].apply(
                lambda x: ecwm.MotionTypes(x).name if x in set(enum.value for enum in ecwm.MotionTypes) else 'UNKNOWN'
            )
        logging.debug(f"Utils - Trajectory Stage 8: Time taken for background/location processing: {time.time() - location_check_start_time:.4f} seconds")

    total_time = time.time() - start_time  # End timer for the entire function
    logging.debug(f"Utils - Trajectory Stage 9: Total time taken for query_trajectories: {total_time:.4f} seconds")

    return df


def add_user_stats(user_data, batch_size=10):
    start_time = time.time()  # Start timer for the entire function
    time_format = 'YYYY-MM-DD HH:mm:ss'

    def process_user(user):
        process_user_start_time = time.time()  # Start timing for process_user
        user_uuid = UUID(user['user_id'])

        # Fetch aggregated data for all users once and cache it
        ts_aggregate_start_time = time.time()
        ts_aggregate = esta.TimeSeries.get_aggregate_time_series()
        logging.debug(f"Utils - User Stats Stage 1: Time taken for get_aggregate_time_series: {time.time() - ts_aggregate_start_time:.4f} seconds")

        # Fetch data for the user, cached for repeated queries
        profile_data_start_time = time.time()
        profile_data = edb.get_profile_db().find_one({'user_id': user_uuid})
        logging.debug(f"Utils - User Stats Stage 2: Time taken for find_one user profile: {time.time() - profile_data_start_time:.4f} seconds")

        # Count total trips and labeled trips
        trips_count_start_time = time.time()
        total_trips = ts_aggregate.find_entries_count(
            key_list=["analysis/confirmed_trip"],
            extra_query_list=[{'user_id': user_uuid}]
        )
        labeled_trips = ts_aggregate.find_entries_count(
            key_list=["analysis/confirmed_trip"],
            extra_query_list=[{'user_id': user_uuid}, {'data.user_input': {'$ne': {}}}]
        )
        logging.debug(f"Utils - User Stats Stage 3: Time taken for counting trips: {time.time() - trips_count_start_time:.4f} seconds")

        user['total_trips'] = total_trips
        user['labeled_trips'] = labeled_trips

        if profile_data:
            user['platform'] = profile_data.get('curr_platform')
            user['manufacturer'] = profile_data.get('manufacturer')
            user['app_version'] = profile_data.get('client_app_version')
            user['os_version'] = profile_data.get('client_os_version')
            user['phone_lang'] = profile_data.get('phone_lang')

        if total_trips > 0:
            ts_user_start_time = time.time()
            ts = esta.TimeSeries.get_time_series(user_uuid)
            logging.debug(f"Utils - User Stats Stage 4: Time taken for get_time_series: {time.time() - ts_user_start_time:.4f} seconds")

            # Get first and last trip timestamps
            first_last_trip_start_time = time.time()
            first_trip_ts = ts.get_first_value_for_field(
                key='analysis/confirmed_trip',
                field='data.end_ts',
                sort_order=pymongo.ASCENDING
            )
            if first_trip_ts != -1:
                user['first_trip'] = arrow.get(first_trip_ts).format(time_format)

            last_trip_ts = ts.get_first_value_for_field(
                key='analysis/confirmed_trip',
                field='data.end_ts',
                sort_order=pymongo.DESCENDING
            )
            if last_trip_ts != -1:
                user['last_trip'] = arrow.get(last_trip_ts).format(time_format)

            last_call_ts = ts.get_first_value_for_field(
                key='stats/server_api_time',
                field='data.ts',
                sort_order=pymongo.DESCENDING
            )
            if last_call_ts != -1:
                user['last_call'] = arrow.get(last_call_ts).format(time_format)
            
            logging.debug(f"Utils - User Stats Stage 5: Time taken for retrieving first/last trip and last call: {time.time() - first_last_trip_start_time:.4f} seconds")

        logging.debug(f"Utils - User Stats Stage 6: Total time taken for process_user: {time.time() - process_user_start_time:.4f} seconds")
        return user

    def batch_process(users_batch):
        batch_start_time = time.time()  # Start timing for batch process
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_user, user) for user in users_batch]
            processed_batch = [future.result() for future in as_completed(futures)]
        logging.debug(f"Utils - User Stats Stage 7: Time taken for batch processing {len(users_batch)} users: {time.time() - batch_start_time:.4f} seconds")
        return processed_batch

    total_users = len(user_data)
    processed_data = []

    for i in range(0, total_users, batch_size):
        batch_start_time = time.time()
        batch = user_data[i:i + batch_size]
        processed_batch = batch_process(batch)
        processed_data.extend(processed_batch)
        logging.debug(f"Processed {len(processed_data)} users out of {total_users}. Time taken for this batch: {time.time() - batch_start_time:.4f} seconds")

    total_time = time.time() - start_time  # End timer for the entire function
    logging.debug(f"Utils - User Stats Stage 8: Total time taken for add_user_stats: {total_time:.4f} seconds")

    return processed_data


def query_segments_crossing_endpoints(poly_region_start, poly_region_end, start_date: str, end_date: str, tz: str, excluded_uuids: list[str]):
    start_time = time.time()  # Start timing for the entire function
    
    # Convert date range to timestamps
    (start_ts, end_ts) = iso_range_to_ts_range(start_date, end_date, tz)
    logging.debug(f"Utils - Segments Stage 1: Converted dates to timestamps: {start_ts}, {end_ts}")

    tq = estt.TimeQuery("data.ts", start_ts, end_ts)
    not_excluded_uuid_query = {'user_id': {'$nin': [UUID(uuid) for uuid in excluded_uuids]}}
    agg_ts = estag.AggregateTimeSeries().get_aggregate_time_series()

    # Query locations matching the start region
    locs_matching_start_start_time = time.time()
    locs_matching_start = agg_ts.get_data_df(
        "analysis/recreated_location",
        geo_query=estg.GeoQuery(['data.loc'], poly_region_start),
        time_query=tq,
        extra_query_list=[not_excluded_uuid_query]
    )
    locs_matching_start = locs_matching_start.drop_duplicates(subset=['section'])
    locs_matching_start_time = time.time() - locs_matching_start_start_time
    logging.debug(f"Utils - Segments Stage 2: Found {len(locs_matching_start)} locations matching the start region in {locs_matching_start_time:.4f} seconds.")
    if locs_matching_start.empty:
        return locs_matching_start
    
    # Query locations matching the end region
    locs_matching_end_start_time = time.time()
    locs_matching_end = agg_ts.get_data_df(
        "analysis/recreated_location",
        geo_query=estg.GeoQuery(['data.loc'], poly_region_end),
        time_query=tq,
        extra_query_list=[not_excluded_uuid_query]
    )
    locs_matching_end = locs_matching_end.drop_duplicates(subset=['section'])
    locs_matching_end_time = time.time() - locs_matching_end_start_time
    logging.debug(f"Utils - Segments Stage 3: Found {len(locs_matching_end)} locations matching the end region in {locs_matching_end_time:.4f} seconds.")
    if locs_matching_end.empty:
        return locs_matching_end
    
    # Merge the two location DataFrames
    merge_start_time = time.time()
    merged = locs_matching_start.merge(locs_matching_end, how='outer', on=['section'])
    merge_time = time.time() - merge_start_time
    logging.debug(f"Utils - Segments Stage 4: Merged DataFrames in {merge_time:.4f} seconds.")
    
    filtered = merged.loc[merged['idx_x'] < merged['idx_y']].copy()
    
    # Calculate duration and format time
    filtered['duration'] = filtered['ts_y'] - filtered['ts_x']
    filtered['mode'] = filtered['mode_x']
    filtered['start_fmt_time'] = filtered['fmt_time_x']
    filtered['end_fmt_time'] = filtered['fmt_time_y']
    filtered['user_id'] = filtered['user_id_y']
    
    number_user_seen = filtered.user_id_x.nunique()
    logging.debug(f"Utils - Segments Stage 5: Number of unique users seen: {number_user_seen}")

    # Check if the number of users meets the minimum requirement
    if perm_utils.permissions.get("segment_trip_time_min_users", 0) <= number_user_seen:
        total_time = time.time() - start_time  # End timing for the entire function
        logging.debug(f"Utils - Segments Stage 6: Total time taken for query_segments_crossing_endpoints: {total_time:.4f} seconds.")
        return filtered
    
    logging.debug("Utils - Segments Stage 7: Minimum user requirement not met, returning empty DataFrame.")
    return pd.DataFrame.from_dict([])


# The following query can be called multiple times, let's open db only once
analysis_timeseries_db = edb.get_analysis_timeseries_db()

# Fetches sensed_mode for each section in a list
# sections format example: [{'section': ObjectId('648d02b227fd2bb6635414a0'), 'user_id': UUID('6d7edf29-8b3f-451b-8d66-984cb8dd8906')}]
def query_inferred_sections_modes(sections):
    return esds.cleaned2inferred_section_list(sections)

