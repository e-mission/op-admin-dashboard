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
import emission.core.timer as ect
import emission.storage.decorations.stats_queries as esdsq

from utils import constants
from utils import permissions as perm_utils
from utils.datetime_utils import iso_range_to_ts_range

def df_to_filtered_records(df, col_to_filter=None, vals_to_exclude=None):
    """
    Filters a DataFrame based on specified column and exclusion values, then converts it to a list of records.

    :param df (pd.DataFrame): The DataFrame to filter.
    :param col_to_filter (str, optional): The column name to apply the filter on.
    :param vals_to_exclude (list[str], optional): List of values to exclude from the filter.
    :return: List of dictionaries representing the filtered records.
    """
    # Stage 1: Validate DataFrame and Set Defaults
    with ect.Timer() as stage1_timer:
        # Check if df is a valid DataFrame and if it is empty
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            # Exiting the context to set 'elapsed_ms'
            pass  # Do nothing here; handle after the 'with' block
        else:
            # Default to an empty list if vals_to_exclude is None
            if vals_to_exclude is None:
                vals_to_exclude = []
    # Store stage1 timing after exiting the 'with' block
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        esdsq.store_dashboard_time(
            "admin/db_utils/df_to_filtered_records/validate_dataframe_and_set_defaults",
            stage1_timer
        )
        return []
    else:
        esdsq.store_dashboard_time(
            "admin/db_utils/df_to_filtered_records/validate_dataframe_and_set_defaults",
            stage1_timer
        )
    
    # Stage 2: Perform Filtering
    with ect.Timer() as stage2_timer:
        # Perform filtering if col_to_filter and vals_to_exclude are provided
        if col_to_filter and vals_to_exclude:
            # Ensure vals_to_exclude is a list of strings
            if not isinstance(vals_to_exclude, list) or not all(isinstance(val, str) for val in vals_to_exclude):
                raise ValueError("vals_to_exclude must be a list of strings.")
            df = df[~df[col_to_filter].isin(vals_to_exclude)]
    # Store stage2 timing after exiting the 'with' block
    esdsq.store_dashboard_time(
        "admin/db_utils/df_to_filtered_records/perform_filtering",
        stage2_timer
    )
    
    # Store total timing
    with ect.Timer() as total_timer:
        pass  # No operations here; 'elapsed_ms' will capture the time from start to now
    esdsq.store_dashboard_time(
        "admin/db_utils/df_to_filtered_records/total_time",
        total_timer
    )
    
    return df.to_dict("records")



def query_uuids(start_date: str, end_date: str, tz: str):
    """
    Queries UUIDs from the database within a specified date range and timezone.

    :param start_date (str): Start date in ISO format.
    :param end_date (str): End date in ISO format.
    :param tz (str): Timezone string.
    :return: Processed pandas DataFrame of UUIDs.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Log Debug Message
        with ect.Timer() as stage1_timer:
            logging.debug("Querying the UUID DB for (no date range)")
        esdsq.store_dashboard_time(
            "admin/db_utils/query_uuids/log_debug_message",
            stage1_timer
        )

        # Stage 2: Fetch Aggregate Time Series
        with ect.Timer() as stage2_timer:
            # This should actually use the profile DB instead of (or in addition to)
            # the UUID DB so that we can see the app version, os, manufacturer...
            # I will write a couple of functions to get all the users in a time range
            # (although we should define what that time range should be) and to merge
            # that with the profile data
            entries = edb.get_uuid_db().find()
            df = pd.json_normalize(list(entries))
        esdsq.store_dashboard_time(
            "admin/db_utils/query_uuids/fetch_aggregate_time_series",
            stage2_timer
        )
        
        # Stage 3: Process DataFrame
        with ect.Timer() as stage3_timer:
            if not df.empty:
                df['update_ts'] = pd.to_datetime(df['update_ts'])
                df['user_id'] = df['uuid'].apply(str)
                df['user_token'] = df['user_email']
                df.drop(columns=["uuid", "_id"], inplace=True)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_uuids/process_dataframe",
            stage3_timer
        )
    
    esdsq.store_dashboard_time(
        "admin/db_utils/query_uuids/total_time",
        total_timer
    )
    
    return df

def query_confirmed_trips(start_date: str, end_date: str, tz: str):
    """
    Queries confirmed trips within a specified date range and timezone.

    :param start_date (str): Start date in ISO format.
    :param end_date (str): End date in ISO format.
    :param tz (str): Timezone string.
    :return: Tuple containing the processed DataFrame and list of user input columns.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Convert Date Range to Timestamps
        with ect.Timer() as stage1_timer:
            (start_ts, end_ts) = iso_range_to_ts_range(start_date, end_date, tz)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_confirmed_trips/convert_date_range_to_timestamps",
            stage1_timer
        )
        
        # Stage 2: Fetch Aggregate Time Series
        with ect.Timer() as stage2_timer:
            ts = esta.TimeSeries.get_aggregate_time_series()
        esdsq.store_dashboard_time(
            "admin/db_utils/query_confirmed_trips/fetch_aggregate_time_series",
            stage2_timer
        )
        
        # Stage 3: Fetch Confirmed Trip Entries
        with ect.Timer() as stage3_timer:
            # Note to self, allow end_ts to also be null in the timequery
            # we can then remove the start_time, end_time logic
            df = ts.get_data_df("analysis/confirmed_trip",
                time_query=estt.TimeQuery("data.start_ts", start_ts, end_ts),
            )
            user_input_cols = []
        esdsq.store_dashboard_time(
            "admin/db_utils/query_confirmed_trips/fetch_confirmed_trip_entries",
            stage3_timer
        )
        
        if not df.empty:
            # Stage 4: Convert Object Columns to Strings
            with ect.Timer() as stage4_timer:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(str)
            esdsq.store_dashboard_time(
                "admin/db_utils/query_confirmed_trips/convert_object_columns_to_strings",
                stage4_timer
            )
            
            # Stage 5: Drop Metadata Columns
            with ect.Timer() as stage5_timer:
                # Drop metadata columns
                columns_to_drop = [col for col in df.columns if col.startswith("metadata")]
                df.drop(columns=columns_to_drop, inplace=True)
            esdsq.store_dashboard_time(
                "admin/db_utils/query_confirmed_trips/drop_metadata_columns",
                stage5_timer
            )
            
            # Stage 6: Drop or Modify Excluded Columns
            with ect.Timer() as stage6_timer:
                # Drop or modify excluded columns
                for col in constants.EXCLUDED_TRAJECTORIES_COLS:
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True)
            esdsq.store_dashboard_time(
                "admin/db_utils/query_confirmed_trips/drop_or_modify_excluded_columns",
                stage6_timer
            )
            
            # I dont think we even implemented this..
            # need fix asap
            # Stage 7: Handle 'background/location' Key
            with ect.Timer() as stage7_timer:
                # Check if 'background/location' is in the key_list
                if 'background/location' in key_list:
                    if 'data.mode' in df.columns:
                        # Set the values in data.mode to blank ('')
                        df['data.mode'] = ''
                else:
                    # Map mode to its corresponding string value
                    df['data.mode_str'] = df['data.mode'].apply(
                        lambda x: ecwm.MotionTypes(x).name if x in set(enum.value for enum in ecwm.MotionTypes) else 'UNKNOWN'
                    )
            esdsq.store_dashboard_time(
                "admin/db_utils/query_confirmed_trips/handle_background_location_key",
                stage7_timer
            )
            
            # Stage 8: Clean and Modify DataFrames
            with ect.Timer() as stage8_timer:
                # Expand the user inputs
                user_input_df = pd.json_normalize(df.user_input)
                df = pd.concat([df, user_input_df], axis='columns')
                user_input_cols = [
                    c for c in user_input_df.columns
                    if "metadata" not in c and
                       "xmlns" not in c and
                       "local_dt" not in c and
                       'xmlResponse' not in c and
                       "_id" not in c
                ]
            esdsq.store_dashboard_time(
                "admin/db_utils/query_confirmed_trips/clean_and_modify_dataframes",
                stage8_timer
            )
            
            # Stage 9: Filter and Combine Columns
            with ect.Timer() as stage9_timer:
                combined_col_list = list(perm_utils.get_all_trip_columns()) + user_input_cols
                columns = [col for col in combined_col_list if col in df.columns]
                df = df[columns]
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
            esdsq.store_dashboard_time(
                "admin/db_utils/query_confirmed_trips/filter_and_combine_columns",
                stage9_timer
            )
        
    esdsq.store_dashboard_time(
        "admin/db_utils/query_confirmed_trips/total_time",
        total_timer
    )
    return (df, user_input_cols)

def query_demographics():
    """
    Queries demographic survey data and organizes it into a dictionary of DataFrames.
    Each key in the dictionary represents a different survey ID, and the corresponding
    value is a DataFrame containing the survey responses.

    :return: Dictionary where keys are survey IDs and values are corresponding DataFrames.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Log Debug Message
        with ect.Timer() as stage1_timer:
            # Returns dictionary of df where key represent different survey id and values are df for each survey
            logging.debug("Querying the demographics for (no date range)")
        esdsq.store_dashboard_time(
            "admin/db_utils/query_demographics/log_debug_message",
            stage1_timer
        )

        # Stage 2: Fetch Aggregate Time Series
        with ect.Timer() as stage2_timer:
            ts = esta.TimeSeries.get_aggregate_time_series()
        esdsq.store_dashboard_time(
            "admin/db_utils/query_demographics/fetch_aggregate_time_series",
            stage2_timer
        )

        # Stage 3: Find Demographic Survey Entries
        with ect.Timer() as stage3_timer:
            entries = ts.find_entries(["manual/demographic_survey"])
            data = list(entries)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_demographics/find_demographic_survey_entries",
            stage3_timer
        )

        # Stage 4: Organize Entries by Survey Key
        with ect.Timer() as stage4_timer:
            available_key = {}
            for entry in data:
                survey_key = list(entry['data']['jsonDocResponse'].keys())[0]
                if survey_key not in available_key:
                    available_key[survey_key] = []
                available_key[survey_key].append(entry)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_demographics/organize_entries_by_survey_key",
            stage4_timer
        )

        # Stage 5: Create DataFrames from Organized Entries
        with ect.Timer() as stage5_timer:
            dataframes = {}
            for key, json_object in available_key.items():
                df = pd.json_normalize(json_object)
                dataframes[key] = df
        esdsq.store_dashboard_time(
            "admin/db_utils/query_demographics/create_dataframes_from_organized_entries",
            stage5_timer
        )

        # Stage 6: Clean and Modify DataFrames
        with ect.Timer() as stage6_timer:
            for key, df in dataframes.items():
                if not df.empty:
                    # Convert binary demographic columns to strings
                    for col in constants.BINARY_DEMOGRAPHICS_COLS:
                        if col in df.columns:
                            df[col] = df[col].apply(str)

                    # Drop metadata columns
                    columns_to_drop = [col for col in df.columns if col.startswith("metadata")]
                    df.drop(columns=columns_to_drop, inplace=True)

                    # Modify column names
                    modified_columns = perm_utils.get_demographic_columns(df.columns)
                    df.columns = modified_columns

                    # Simplify column names by removing prefixes
                    df.columns = [
                        col.rsplit('.', 1)[-1] if col.startswith('data.jsonDocResponse.') else col
                        for col in df.columns
                    ]

                    # Drop excluded demographic columns
                    for col in constants.EXCLUDED_DEMOGRAPHICS_COLS:
                        if col in df.columns:
                            df.drop(columns=[col], inplace=True)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_demographics/clean_and_modify_dataframes",
            stage6_timer
        )

    esdsq.store_dashboard_time(
        "admin/db_utils/query_demographics/total_time",
        total_timer
    )

    return dataframes

def query_trajectories(start_date: str, end_date: str, tz: str, key_list: list[str]):
    """
    Queries trajectories within a specified date range and timezone based on provided key list.
    
    :param start_date (str): Start date in ISO format.
    :param end_date (str): End date in ISO format.
    :param tz (str): Timezone string.
    :param key_list (list[str]): List of keys to query.
    :return: Processed pandas DataFrame of trajectories.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Convert Date Range to Timestamps
        with ect.Timer() as stage1_timer:
            (start_ts, end_ts) = iso_range_to_ts_range(start_date, end_date, tz)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_trajectories/convert_date_range_to_timestamps",
            stage1_timer
        )
        
        # Stage 2: Fetch Aggregate Time Series
        with ect.Timer() as stage2_timer:
            ts = esta.TimeSeries.get_aggregate_time_series()
        esdsq.store_dashboard_time(
            "admin/db_utils/query_trajectories/fetch_aggregate_time_series",
            stage2_timer
        )
        
        # Stage 3: Fetch Trajectory Entries
        with ect.Timer() as stage3_timer:
            # Check if key_list contains 'background/location'
            key_list = [key_list]
            entries = ts.find_entries(
                key_list=key_list,
                time_query=estt.TimeQuery("data.ts", start_ts, end_ts),
            )
            df = pd.json_normalize(list(entries))
        esdsq.store_dashboard_time(
            "admin/db_utils/query_trajectories/fetch_trajectory_entries",
            stage3_timer
        )
        
        if not df.empty:
            # Stage 4: Convert Object Columns to Strings
            with ect.Timer() as stage4_timer:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(str)
            esdsq.store_dashboard_time(
                "admin/db_utils/query_trajectories/convert_object_columns_to_strings",
                stage4_timer
            )
            
            # Stage 5: Drop Metadata Columns
            with ect.Timer() as stage5_timer:
                # Drop metadata columns
                columns_to_drop = [col for col in df.columns if col.startswith("metadata")]
                df.drop(columns=columns_to_drop, inplace=True)
            esdsq.store_dashboard_time(
                "admin/db_utils/query_trajectories/drop_metadata_columns",
                stage5_timer
            )
            
            # Stage 6: Drop or Modify Excluded Columns
            with ect.Timer() as stage6_timer:
                # Drop or modify excluded columns
                for col in constants.EXCLUDED_TRAJECTORIES_COLS:
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True)
            esdsq.store_dashboard_time(
                "admin/db_utils/query_trajectories/drop_or_modify_excluded_columns",
                stage6_timer
            )
            
            # Stage 7: Handle 'background/location' Key
            with ect.Timer() as stage7_timer:
                # Check if 'background/location' is in the key_list
                if 'background/location' in key_list:
                    if 'data.mode' in df.columns:
                        # Set the values in data.mode to blank ('')
                        df['data.mode'] = ''
                else:
                    # Map mode to its corresponding string value
                    df['data.mode_str'] = df['data.mode'].apply(
                        lambda x: ecwm.MotionTypes(x).name if x in set(enum.value for enum in ecwm.MotionTypes) else 'UNKNOWN'
                    )
            esdsq.store_dashboard_time(
                "admin/db_utils/query_trajectories/handle_background_location_key",
                stage7_timer
            )
        
    esdsq.store_dashboard_time(
        "admin/db_utils/query_trajectories/total_time",
        total_timer
    )
    return df

def add_user_stats(user_data):
    """
    Adds statistical data to each user in the provided user_data list.

    For each user, it calculates total trips, labeled trips, and retrieves profile information.
    Additionally, it records the timestamps of the first trip, last trip, and the last API call.

    :param user_data (list[dict]): List of user dictionaries to be enriched with stats.
    :return: The list of user dictionaries with added statistical data.
    """
    with ect.Timer(verbose=False) as total_timer:
        logging.info("Adding user stats")
        
        for user in user_data:
            with ect.Timer(verbose=False) as stage_timer:
                try:
                    logging.debug(f"Processing user {user['user_id']}")
                    user_uuid = UUID(user['user_id'])

                    # Stage 1: Calculate Total Trips
                    total_trips = esta.TimeSeries.get_aggregate_time_series().find_entries_count(
                        key_list=["analysis/confirmed_trip"],
                        extra_query_list=[{'user_id': user_uuid}]
                    )
                    user['total_trips'] = total_trips

                    # Stage 2: Calculate Labeled Trips
                    labeled_trips = esta.TimeSeries.get_aggregate_time_series().find_entries_count(
                        key_list=["analysis/confirmed_trip"],
                        extra_query_list=[{'user_id': user_uuid}, {'data.user_input': {'$ne': {}}}]
                    )
                    user['labeled_trips'] = labeled_trips

                    # Stage 3: Retrieve Profile Data
                    profile_data = edb.get_profile_db().find_one({'user_id': user_uuid})
                    user['platform'] = profile_data.get('curr_platform')
                    user['manufacturer'] = profile_data.get('manufacturer')
                    user['app_version'] = profile_data.get('client_app_version')
                    user['os_version'] = profile_data.get('client_os_version')
                    user['phone_lang'] = profile_data.get('phone_lang')

                    if total_trips > 0:
                        time_format = 'YYYY-MM-DD HH:mm:ss'
                        ts = esta.TimeSeries.get_time_series(user_uuid)
                        
                        # Stage 4: Get First Trip Timestamp
                        start_ts = ts.get_first_value_for_field(
                            key='analysis/confirmed_trip',
                            field='data.end_ts',
                            sort_order=pymongo.ASCENDING
                        )
                        if start_ts != -1:
                            user['first_trip'] = arrow.get(start_ts).format(time_format)

                        # Stage 5: Get Last Trip Timestamp
                        end_ts = ts.get_first_value_for_field(
                            key='analysis/confirmed_trip',
                            field='data.end_ts',
                            sort_order=pymongo.DESCENDING
                        )
                        if end_ts != -1:
                            user['last_trip'] = arrow.get(end_ts).format(time_format)

                        # Stage 6: Get Last API Call Timestamp
                        last_call = ts.get_first_value_for_field(
                            key='stats/server_api_time',
                            field='data.ts',
                            sort_order=pymongo.DESCENDING
                        )
                        if last_call != -1:
                            user['last_call'] = arrow.get(last_call).format(time_format)

                except Exception as e:
                    logging.exception(f"An error occurred while processing user {user.get('user_id', 'Unknown')}: {e}")
                finally:
                    # Store timing for processing each user
                    # I'm hesistant to store this because it will be a lot of data
                    # esdsq.store_dashboard_time(
                    #     f"admin/db_utils/add_user_stats/process_user_{user['user_id']}",
                    #     stage_timer  # Pass the Timer object
                    # )
                    pass
        
        logging.info("Finished adding user stats")
    
    # Store total timing for the entire function
    esdsq.store_dashboard_time(
        "admin/db_utils/add_user_stats/total_time",
        total_timer  # Pass the Timer object
    )
    
    return user_data

def query_segments_crossing_endpoints(
    poly_region_start,
    poly_region_end,
    start_date: str,
    end_date: str,
    tz: str,
    excluded_uuids: list[str]
):
    """
    Queries segments that cross specified start and end polygon regions within a given date range,
    excluding specified user UUIDs. Returns a DataFrame of filtered segments that meet the criteria.

    :param poly_region_start: Polygon defining the start region.
    :param poly_region_end: Polygon defining the end region.
    :param start_date (str): Start date in ISO format.
    :param end_date (str): End date in ISO format.
    :param tz (str): Timezone string.
    :param excluded_uuids (list[str]): List of user UUIDs to exclude.
    :return: Filtered pandas DataFrame of segments crossing the endpoints.
    """
    with ect.Timer() as total_timer:
        # Stage 1: Convert Date Range to Timestamps
        with ect.Timer() as stage1_timer:
            start_ts, end_ts = iso_range_to_ts_range(start_date, end_date, tz)
        esdsq.store_dashboard_time(
            "admin/db_utils/query_segments_crossing_endpoints/convert_date_range_to_timestamps",
            stage1_timer
        )
        
        # Stage 2: Setup Time and User Exclusion Queries
        with ect.Timer() as stage2_timer:
            tq = estt.TimeQuery("data.ts", start_ts, end_ts)
            not_excluded_uuid_query = {
                'user_id': {'$nin': [UUID(uuid) for uuid in excluded_uuids]}
            }
            agg_ts = estag.AggregateTimeSeries().get_aggregate_time_series()
        esdsq.store_dashboard_time(
            "admin/db_utils/query_segments_crossing_endpoints/setup_time_and_user_exclusion_queries",
            stage2_timer
        )

        # Stage 3: Fetch Locations Matching Start Region
        with ect.Timer() as stage3_timer:
            locs_matching_start = agg_ts.get_data_df(
                "analysis/recreated_location",
                geo_query=estg.GeoQuery(['data.loc'], poly_region_start),
                time_query=tq,
                extra_query_list=[not_excluded_uuid_query]
            )
            locs_matching_start = locs_matching_start.drop_duplicates(subset=['section'])
            if locs_matching_start.empty:
                esdsq.store_dashboard_time(
                    "admin/db_utils/query_segments_crossing_endpoints/fetch_locations_matching_start_region",
                    stage3_timer
                )
                esdsq.store_dashboard_time(
                    "admin/db_utils/query_segments_crossing_endpoints/total_time",
                    total_timer
                )
                return locs_matching_start
        esdsq.store_dashboard_time(
            "admin/db_utils/query_segments_crossing_endpoints/fetch_locations_matching_start_region",
            stage3_timer
        )

        # Stage 4: Fetch Locations Matching End Region
        with ect.Timer() as stage4_timer:
            locs_matching_end = agg_ts.get_data_df(
                "analysis/recreated_location",
                geo_query=estg.GeoQuery(['data.loc'], poly_region_end),
                time_query=tq,
                extra_query_list=[not_excluded_uuid_query]
            )
            locs_matching_end = locs_matching_end.drop_duplicates(subset=['section'])
            if locs_matching_end.empty:
                esdsq.store_dashboard_time(
                    "admin/db_utils/query_segments_crossing_endpoints/fetch_locations_matching_end_region",
                    stage4_timer
                )
                esdsq.store_dashboard_time(
                    "admin/db_utils/query_segments_crossing_endpoints/total_time",
                    total_timer
                )
                return locs_matching_end
        esdsq.store_dashboard_time(
            "admin/db_utils/query_segments_crossing_endpoints/fetch_locations_matching_end_region",
            stage4_timer
        )

        # Stage 5: Merge and Filter Segments
        with ect.Timer() as stage5_timer:
            merged = locs_matching_start.merge(
                locs_matching_end, how='outer', on=['section']
            )
            filtered = merged.loc[merged['idx_x'] < merged['idx_y']].copy()
            filtered['duration'] = filtered['ts_y'] - filtered['ts_x']
            filtered['mode'] = filtered['mode_x']
            filtered['start_fmt_time'] = filtered['fmt_time_x']
            filtered['end_fmt_time'] = filtered['fmt_time_y']
            filtered['user_id'] = filtered['user_id_y']
        esdsq.store_dashboard_time(
            "admin/db_utils/query_segments_crossing_endpoints/merge_and_filter_segments",
            stage5_timer
        )

        # Stage 6: Evaluate User Count and Final Filtering
        with ect.Timer() as stage6_timer:
            number_user_seen = filtered.user_id_x.nunique()
            min_users_required = perm_utils.permissions.get(
                "segment_trip_time_min_users", 0
            )
            logging.debug(
                f"Number of unique users seen: {number_user_seen} "
                f"(Minimum required: {min_users_required})"
            )
            if number_user_seen >= min_users_required:
                logging.info(
                    f"Returning filtered segments with {number_user_seen} unique users."
                )
                result = filtered
            else:
                logging.info(
                    f"Insufficient unique users ({number_user_seen}) to meet the "
                    f"minimum requirement ({min_users_required}). Returning empty DataFrame."
                )
                result = pd.DataFrame.from_dict([])
        esdsq.store_dashboard_time(
            "admin/db_utils/query_segments_crossing_endpoints/evaluate_user_count_and_final_filtering",
            stage6_timer
        )

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

