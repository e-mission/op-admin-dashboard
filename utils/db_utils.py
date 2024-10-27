import logging
import arrow
from uuid import UUID

import pandas as pd
import pymongo

import emission.core.get_database as edb
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.timeseries.aggregate_timeseries as estag
import emission.storage.timeseries.timequery as estt
import emission.core.wrapper.motionactivity as ecwm
import emission.storage.timeseries.geoquery as estg
import emission.storage.decorations.section_queries as esds
import emission.core.timer as ect
import emission.storage.decorations.stats_queries as esdsq

from utils import constants
from utils import permissions as perm_utils
from utils.datetime_utils import iso_range_to_ts_range
from concurrent.futures import ThreadPoolExecutor, as_completed

def df_to_filtered_records(df, col_to_filter=None, vals_to_exclude: list[str] = []):
    """
    Returns a dictionary of df records, given a dataframe, a column to filter on,
    and a list of values that rows in that column will be excluded if they match
    """
    with ect.Timer() as total_timer:

        if df.empty:
            return []

        # Stage 2: Filter DataFrame if needed
        with ect.Timer() as stage2_timer:
            if col_to_filter and vals_to_exclude:  # will only filter if both are not None or []
                df = df[~df[col_to_filter].isin(vals_to_exclude)]
        esdsq.store_dashboard_time(
            "admin/db_utils/df_to_filtered_records/filter_dataframe_if_needed",
            stage2_timer
        )

        # Stage 3: Convert DataFrame to dict of records
        with ect.Timer() as stage3_timer:
            result = df.to_dict("records")
        esdsq.store_dashboard_time(
            "admin/db_utils/df_to_filtered_records/convert_to_dict_of_records",
            stage3_timer
        )

    esdsq.store_dashboard_time(
        "admin/db_utils/df_to_filtered_records/total_time",
        total_timer
    )

    return result


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
    with ect.Timer() as total_timer:

        # Stage 1: Logging and query initiation
        with ect.Timer() as stage1_timer:
            logging.debug("Querying the UUID DB for (no date range)")
            entries = edb.get_uuid_db().find()
        esdsq.store_dashboard_time(
            "admin/db_utils/query_uuids/logging_and_query_initiation",
            stage1_timer
        )

        # Stage 2: Convert query result to DataFrame
        with ect.Timer() as stage2_timer:
            df = pd.json_normalize(list(entries))
        esdsq.store_dashboard_time(
            "admin/db_utils/query_uuids/convert_query_result_to_dataframe",
            stage2_timer
        )

        # Stage 3: DataFrame processing
        with ect.Timer() as stage3_timer:
            if not df.empty:
                df['update_ts'] = pd.to_datetime(df['update_ts'])
                df['user_id'] = df['uuid'].apply(str)
                df['user_token'] = df['user_email']
                df.drop(columns=["uuid", "_id"], inplace=True)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_uuids/dataframe_processing",
            stage3_timer
        )

    esdsq.store_dashboard_time(
        "admin/db_utils/query_uuids/total_time",
        total_timer
    )

    return df

def query_confirmed_trips(start_date: str, end_date: str, tz: str):
    with ect.Timer() as total_timer:

        # Stage 1: Convert date range to timestamps
        with ect.Timer() as stage1_timer:
            (start_ts, end_ts) = iso_range_to_ts_range(start_date, end_date, tz)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_confirmed_trips/convert_date_range_to_timestamps",
            stage1_timer
        )

        # Stage 2: Retrieve aggregate time series
        with ect.Timer() as stage2_timer:
            ts = esta.TimeSeries.get_aggregate_time_series()
            # Note to self, allow end_ts to also be null in the timequery
            # we can then remove the start_time, end_time logic
            df = ts.get_data_df(
                "analysis/confirmed_trip",
                time_query=estt.TimeQuery("data.start_ts", start_ts, end_ts),
            )
        esdsq.store_dashboard_time(
            "admin/db_utils/query_confirmed_trips/retrieve_aggregate_time_series",
            stage2_timer
        )

        user_input_cols = []

        logging.debug("Before filtering, df columns are %s" % df.columns)
        if not df.empty:
            # Since we use `get_data_df` instead of `pd.json_normalize`,
            # we lose the "data" prefix on the fields and they are only flattened one level
            # Here, we restore the prefix for the VALID_TRIP_COLS from constants.py
            # for backwards compatibility. We do this for all columns since columns which don't exist are ignored by the rename command.
            
            # Stage 3: Rename columns
            with ect.Timer() as stage3_timer:
                rename_cols = constants.VALID_TRIP_COLS
                # the mapping is `{distance: data.distance, duration: data.duration} etc
                rename_mapping = dict(zip([c.replace("data.", "") for c in rename_cols], rename_cols))
                logging.debug("Rename mapping is %s" % rename_mapping)
                df.rename(columns=rename_mapping, inplace=True)
                logging.debug("After renaming columns, they are %s" % df.columns)
            esdsq.store_dashboard_time(
                "admin/db_utils/query_confirmed_trips/rename_columns",
                stage3_timer
            )

            # Stage 4: Process coordinates and modes
            with ect.Timer() as stage4_timer:
                # Now copy over the coordinates
                df['data.start_loc.coordinates'] = df['start_loc'].apply(lambda g: g["coordinates"])
                df['data.end_loc.coordinates'] = df['end_loc'].apply(lambda g: g["coordinates"])

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
                    # If 'md' is a dictionary and 'distance' is a valid key pointing to a dictionary:
                    else (
                        # Get the maximum value from 'md["distance"]' using the values of 'md["distance"].get' as the key for 'max'.
                        # This operation only happens if the length of 'md["distance"]' is greater than 0.
                        # Otherwise, return "INVALID".
                        max(md["distance"], key=md["distance"].get)
                        if len(md["distance"]) > 0
                        else "INVALID"
                    )
                )
                df["data.primary_sensed_mode"] = df.cleaned_section_summary.apply(get_max_mode_from_summary)
                df["data.primary_predicted_mode"] = df.inferred_section_summary.apply(get_max_mode_from_summary)
                if 'ble_sensed_summary' in df.columns:
                    df["data.primary_ble_sensed_mode"] = df.ble_sensed_summary.apply(get_max_mode_from_summary)
                else:
                    logging.debug("No BLE support found, not fleet version, ignoring...")
            esdsq.store_dashboard_time(
                "admin/db_utils/query_confirmed_trips/process_coordinates_and_modes",
                stage4_timer
            )

            # Stage 5: Expand user inputs and filter columns
            with ect.Timer() as stage5_timer:
                # Expand the user inputs
                user_input_df = pd.json_normalize(df.user_input)
                df = pd.concat([df, user_input_df], axis='columns')
                logging.debug(f"Before filtering {user_input_df.columns=}")
                user_input_cols = [c for c in user_input_df.columns
                    if "metadata" not in c and
                       "xmlns" not in c and
                       "local_dt" not in c and
                       'xmlResponse' not in c and
                       "_id" not in c]
                logging.debug(f"After filtering {user_input_cols=}")

                combined_col_list = list(perm_utils.get_all_trip_columns()) + user_input_cols
                logging.debug(f"Combined list {combined_col_list=}")
                columns = [col for col in combined_col_list if col in df.columns]
                df = df[columns]
                logging.debug(f"After filtering against the combined list {df.columns=}")
            esdsq.store_dashboard_time(
                "admin/db_utils/query_confirmed_trips/expand_and_filter_user_inputs",
                stage5_timer
            )

            # Stage 6: Convert binary trip columns and process named columns
            with ect.Timer() as stage6_timer:
                # logging.debug("After getting all columns, they are %s" % df.columns)
                for col in constants.BINARY_TRIP_COLS:
                    if col in df.columns:
                        df[col] = df[col].apply(str)
                for named_col in perm_utils.get_all_named_trip_columns():
                    if named_col['path'] in df.columns:
                        df[named_col['label']] = df[named_col['path']]
                        # df = df.drop(columns=[named_col['path']])
            esdsq.store_dashboard_time(
                "admin/db_utils/query_confirmed_trips/process_binary_and_named_columns",
                stage6_timer
            )

            # Stage 7: Humanize distance and duration values
            with ect.Timer() as stage7_timer:
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
            esdsq.store_dashboard_time(
                "admin/db_utils/query_confirmed_trips/humanize_distance_and_duration",
                stage7_timer
            )

    esdsq.store_dashboard_time(
        "admin/db_utils/query_confirmed_trips/total_time",
        total_timer
    )

    return (df, user_input_cols)


def query_demographics():
    with ect.Timer() as total_timer:

        # Stage 1: Query demographics data
        with ect.Timer() as stage1_timer:
            # Returns dictionary of df where key represent different survey id and values are df for each survey
            logging.debug("Querying the demographics for (no date range)")
            ts = esta.TimeSeries.get_aggregate_time_series()
            entries = ts.find_entries(["manual/demographic_survey"])
            data = list(entries)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_demographics/query_data",
            stage1_timer
        )

        # Stage 2: Organize survey keys
        with ect.Timer() as stage2_timer:
            available_key = {}
            for entry in data:
                survey_key = list(entry['data']['jsonDocResponse'].keys())[0]
                if survey_key not in available_key:
                    available_key[survey_key] = []
                available_key[survey_key].append(entry)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_demographics/organize_survey_keys",
            stage2_timer
        )

        # Stage 3: Create dataframes for each survey key
        with ect.Timer() as stage3_timer:
            dataframes = {}
            for key, json_object in available_key.items():
                df = pd.json_normalize(json_object)
                dataframes[key] = df
        esdsq.store_dashboard_time(
            "admin/db_utils/query_demographics/create_dataframes",
            stage3_timer
        )

        # Stage 4: Process each dataframe
        with ect.Timer() as stage4_timer:
            for key, df in dataframes.items():
                if not df.empty:
                    # Convert binary demographic columns
                    for col in constants.BINARY_DEMOGRAPHICS_COLS:
                        if col in df.columns:
                            df[col] = df[col].apply(str) 
                    
                    # Drop metadata columns
                    columns_to_drop = [col for col in df.columns if col.startswith("metadata")]
                    df.drop(columns=columns_to_drop, inplace=True)

                    # Modify columns based on demographic settings
                    modified_columns = perm_utils.get_demographic_columns(df.columns)  
                    df.columns = modified_columns 

                    # Simplify column names for display
                    df.columns = [col.rsplit('.', 1)[-1] if col.startswith('data.jsonDocResponse.') else col for col in df.columns]  

                    # Drop excluded demographic columns
                    for col in constants.EXCLUDED_DEMOGRAPHICS_COLS:
                        if col in df.columns:
                            df.drop(columns=[col], inplace=True)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_demographics/process_dataframes",
            stage4_timer
        )

    esdsq.store_dashboard_time(
        "admin/db_utils/query_demographics/total_time",
        total_timer
    )

    return dataframes


def query_trajectories(start_date: str, end_date: str, tz: str, key_list):
    with ect.Timer() as total_timer:
        key_list = [key_list] if isinstance(key_list, str) else key_list
        # Stage 1: Convert date range to timestamps
        with ect.Timer() as stage1_timer:
            (start_ts, end_ts) = iso_range_to_ts_range(start_date, end_date, tz)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_trajectories/convert_date_range_to_timestamps",
            stage1_timer
        )

        # Stage 2: Retrieve entries from the time series
        with ect.Timer() as stage2_timer:
            ts = esta.TimeSeries.get_aggregate_time_series()
            entries = ts.find_entries(
                key_list=key_list,
                time_query=estt.TimeQuery("data.ts", start_ts, end_ts),
            )
        esdsq.store_dashboard_time(
            "admin/db_utils/query_trajectories/retrieve_entries",
            stage2_timer
        )

        # Stage 3: Convert entries to DataFrame
        with ect.Timer() as stage3_timer:
            df = pd.json_normalize(list(entries))
        esdsq.store_dashboard_time(
            "admin/db_utils/query_trajectories/convert_entries_to_dataframe",
            stage3_timer
        )

        if not df.empty:
            # Stage 4: Process DataFrame columns (convert objects, drop metadata)
            with ect.Timer() as stage4_timer:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(str)

                # Drop metadata columns
                columns_to_drop = [col for col in df.columns if col.startswith("metadata")]
                df.drop(columns=columns_to_drop, inplace=True)

                # Drop excluded trajectory columns
                for col in constants.EXCLUDED_TRAJECTORIES_COLS:
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True)
            esdsq.store_dashboard_time(
                "admin/db_utils/query_trajectories/process_dataframe_columns",
                stage4_timer
            )

            # Stage 5: Add human-readable mode string
            with ect.Timer() as stage5_timer:
                if 'background/location' in key_list:
                    if 'data.mode' in df.columns:
                        # Set the values in data.mode to blank ('')
                        df['data.mode'] = ''
                else:
                    df['data.mode_str'] = df['data.mode'].apply(
                        lambda x: ecwm.MotionTypes(x).name if x in set(enum.value for enum in ecwm.MotionTypes) else 'UNKNOWN'
                    )
            esdsq.store_dashboard_time(
                "admin/db_utils/query_trajectories/add_mode_string",
                stage5_timer
            )

    esdsq.store_dashboard_time(
        "admin/db_utils/query_trajectories/total_time",
        total_timer
    )

    return df



def add_user_stats(user_data, batch_size=5):
    time_format = 'YYYY-MM-DD HH:mm:ss'
    def process_user(user):
        user_uuid = UUID(user['user_id'])
        
        # Fetch aggregated data for all users once and cache it
        ts_aggregate = esta.TimeSeries.get_aggregate_time_series()

        # Fetch data for the user, cached for repeated queries
        profile_data = edb.get_profile_db().find_one({'user_id': user_uuid})
        
        total_trips = ts_aggregate.find_entries_count(
            key_list=["analysis/confirmed_trip"],
            extra_query_list=[{'user_id': user_uuid}]
        )
        labeled_trips = ts_aggregate.find_entries_count(
            key_list=["analysis/confirmed_trip"],
            extra_query_list=[{'user_id': user_uuid}, {'data.user_input': {'$ne': {}}}]
        )
        
        user['total_trips'] = total_trips
        user['labeled_trips'] = labeled_trips

        if profile_data:
            user['platform'] = profile_data.get('curr_platform')
            user['manufacturer'] = profile_data.get('manufacturer')
            user['app_version'] = profile_data.get('client_app_version')
            user['os_version'] = profile_data.get('client_os_version')
            user['phone_lang'] = profile_data.get('phone_lang')

        if total_trips > 0:
            ts = esta.TimeSeries.get_time_series(user_uuid)
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
        
        return user

    def batch_process(users_batch):
        with ThreadPoolExecutor() as executor:  # Adjust max_workers based on CPU cores
            futures = [executor.submit(process_user, user) for user in users_batch]
            processed_batch = [future.result() for future in as_completed(futures)]
        return processed_batch

    total_users = len(user_data)
    processed_data = []

    for i in range(0, total_users, batch_size):
        batch = user_data[i:i + batch_size]
        processed_batch = batch_process(batch)
        processed_data.extend(processed_batch)

        logging.debug(f'Processed {len(processed_data)} users out of {total_users}')

    return processed_data

def query_segments_crossing_endpoints(poly_region_start, poly_region_end, start_date: str, end_date: str, tz: str, excluded_uuids: list[str]):
    with ect.Timer() as total_timer:

        # Stage 1: Convert date range to timestamps and set up time and exclusion queries
        with ect.Timer() as stage1_timer:
            (start_ts, end_ts) = iso_range_to_ts_range(start_date, end_date, tz)
            tq = estt.TimeQuery("data.ts", start_ts, end_ts)
            not_excluded_uuid_query = {'user_id': {'$nin': [UUID(uuid) for uuid in excluded_uuids]}}
            agg_ts = estag.AggregateTimeSeries().get_aggregate_time_series()
        esdsq.store_dashboard_time(
            "admin/db_utils/query_segments_crossing_endpoints/convert_date_range_and_setup_queries",
            stage1_timer
        )

        # Stage 2: Fetch locations matching start region
        with ect.Timer() as stage2_timer:
            locs_matching_start = agg_ts.get_data_df(
                                      "analysis/recreated_location",
                                      geo_query=estg.GeoQuery(['data.loc'], poly_region_start),
                                      time_query=tq,
                                      extra_query_list=[not_excluded_uuid_query]
                                  )
            locs_matching_start = locs_matching_start.drop_duplicates(subset=['section'])
        esdsq.store_dashboard_time(
            "admin/db_utils/query_segments_crossing_endpoints/fetch_start_locations",
            stage2_timer
        )

        # Stage 3: Fetch locations matching end region
        with ect.Timer() as stage3_timer:
            locs_matching_end = agg_ts.get_data_df(
                                    "analysis/recreated_location",
                                    geo_query=estg.GeoQuery(['data.loc'], poly_region_end),
                                    time_query=tq,
                                    extra_query_list=[not_excluded_uuid_query]
                                )
            locs_matching_end = locs_matching_end.drop_duplicates(subset=['section'])
        esdsq.store_dashboard_time(
            "admin/db_utils/query_segments_crossing_endpoints/fetch_end_locations",
            stage3_timer
        )

        # Early returns in case of empty results are handled after timers exit
        if locs_matching_start.empty:
            result = locs_matching_start
        elif locs_matching_end.empty:
            result = locs_matching_end
        else:
            # Stage 4: Merge start and end locations
            with ect.Timer() as stage4_timer:
                merged = locs_matching_start.merge(locs_matching_end, how='outer', on=['section'])
                filtered = merged.loc[merged['idx_x'] < merged['idx_y']].copy()
                filtered['duration'] = filtered['ts_y'] - filtered['ts_x']
                filtered['mode'] = filtered['mode_x']
                filtered['start_fmt_time'] = filtered['fmt_time_x']
                filtered['end_fmt_time'] = filtered['fmt_time_y']
                filtered['user_id'] = filtered['user_id_y']
            esdsq.store_dashboard_time(
                "admin/db_utils/query_segments_crossing_endpoints/merge_and_filter_segments",
                stage4_timer
            )

            # Stage 5: Evaluate user count and determine if results should be returned
            with ect.Timer() as stage5_timer:
                number_user_seen = filtered.user_id_x.nunique()
                if perm_utils.permissions.get("segment_trip_time_min_users", 0) <= number_user_seen:
                    result = filtered
                else:
                    result = pd.DataFrame.from_dict([])
            esdsq.store_dashboard_time(
                "admin/db_utils/query_segments_crossing_endpoints/evaluate_user_count",
                stage5_timer
            )

    # Ensure total time is always logged and the Timer exits before returning the result
    esdsq.store_dashboard_time(
        "admin/db_utils/query_segments_crossing_endpoints/total_time",
        total_timer
    )

    return result


# The following query can be called multiple times, let's open db only once
analysis_timeseries_db = edb.get_analysis_timeseries_db()

# Fetches sensed_mode for each section in a list
# sections format example: [{'section': ObjectId('648d02b227fd2bb6635414a0'), 'user_id': UUID('6d7edf29-8b3f-451b-8d66-984cb8dd8906')}]
def query_inferred_sections_modes(sections):
    return esds.cleaned2inferred_section_list(sections)

