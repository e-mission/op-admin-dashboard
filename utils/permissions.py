import json
import os

import requests
import logging

from utils import constants

STUDY_CONFIG = os.getenv('STUDY_CONFIG')
PATH = os.getenv('CONFIG_PATH')
CONFIG_URL = PATH + STUDY_CONFIG + ".nrel-op.json"
response = requests.get(CONFIG_URL)
config = json.loads(response.text)
surveyinfo = config.get("survey_info",
    {
      "surveys": {
        "UserProfileSurvey": {
          "formPath": "json/demo-survey-v2.json",
          "version": 1,
          "compatibleWith": 1,
          "dataKey": "manual/demographic_survey",
          "labelTemplate": {
            "en": "Answered",
            "es": "Contestada"
          }
        }
      },
      "trip-labels": "MULTILABEL"
    })
permissions = config.get("admin_dashboard", {})

# TODO: The current dynamic config does not have the data_demographics_columns_exclude.
# When all the current studies are completed we can remove the below changes.
if 'data_demographics_columns_exclude' not in permissions:
    permissions['data_demographics_columns_exclude'] = []
if 'data_trajectories_columns_exclude' not in permissions:
    permissions['data_trajectories_columns_exclude'] = []

def has_permission(perm):
    return False if permissions.get(perm) is False else True


def get_allowed_named_trip_columns():
    if surveyinfo["trip-labels"] == "MULTILABEL":
        return constants.MULTILABEL_NAMED_COLS
    elif surveyinfo["trip-labels"] == "ENKETO":
        # TODO: Figure out how to specify these
        # can we re-use the existing labels in survey_info
        # if not, we should add the label paths to survey info
        # since the paths are survey info and not permissions
        # we should also make sure that there are sufficient examples
        # of this
        return permissions.get('additional_trip_columns', [])

def get_required_columns():
    required_cols = {'user_id'}
    required_cols.update(col['path'] for col in constants.REQUIRED_NAMED_COLS)
    return required_cols


def get_all_named_trip_columns():
    named_columns = [item for item in constants.REQUIRED_NAMED_COLS]
    named_columns.extend(
        get_allowed_named_trip_columns()
    )
    return named_columns

def get_all_trip_columns():
    columns = set()
    # logging.debug("get_all_trip_columns: curr set is %s" % columns)
    columns.update(get_allowed_trip_columns())
    # logging.debug("get_all_trip_columns: curr set is %s" % columns)
    columns.update(
        col['path'] for col in get_allowed_named_trip_columns()
    )
    # logging.debug("get_all_trip_columns: curr set is %s" % columns)

    columns.update(get_required_columns())
    # logging.debug("get_all_trip_columns: curr set is %s" % columns)
    columns.update(permissions.get('additional_trip_columns', []))
    logging.debug("get_all_trip_columns: after additional columns, curr set is %s" % columns)
    return columns


def get_allowed_trip_columns():
    columns = set(constants.VALID_TRIP_COLS)
    for column in permissions.get("data_trips_columns_exclude", []):
        columns.discard(column)
    for column in permissions.get("additional_trip_columns", []):
        columns.add(column)
    logging.debug("allowed trip columns are %s" % columns)
    return columns


def get_uuids_columns():
    columns = set(constants.VALID_UUIDS_COLS)
    for column in permissions.get("data_uuids_columns_exclude", []):
        columns.discard(column)
    return columns

def get_demographic_columns(columns):
    for column in permissions.get("data_demographics_columns_exclude", []):
        columns.discard(column)
    return columns

def get_trajectories_columns(columns):
    columns = set(columns)
    for column in permissions.get("data_trajectories_columns_exclude", []):
        columns.discard(column)
    return columns

def get_token_prefix():
    return permissions.get('token_prefix', 'nrelop')