# Dockerized Dash App Template


https://towardsdatascience.com/dockerize-your-dash-app-1e155dd1cea3

## How to run it: Docker Compose (recommended)

`docker compose -f docker-compose-dev.yml build`

`docker compose -f docker-compose-dev.yml up`

You **must** use this method. Do **not** try to directly by setting up a virtualenv with the `requirements.txt`
This uses components of the e-mission-server core, so it must have the e-mission-server modules in the PYTHONPATH
The easiest method to accomplish this is to use the docker container, which is built on top of the e-mission-server base docker container.

## Test with a reverse proxy

```
docker-compose -f docker-compose-prod-nginx.yml build
docker-compose -f docker-compose-prod-nginx.yml up -d
```

The dash component assumes that it is running from the host root, and will barf if you try to run it behind a reverse proxy with a different file prefix.
This setting tests that configuration using an embedded reverse proxy in the docker container.

# Dynamic Config

## Set Variables

### CONFIG_PATH

The `CONFIG_PATH` environment variable is used to specify the location of the configuration files that are required for
a Docker container to run properly. This means that the Docker container will attempt to download the configuration
files from the specified URL. The current path to the raw format of nrel configs is:

https://raw.githubusercontent.com/e-mission/nrel-openpath-deploy-configs/main/configs/


### STUDY_CONFIG

The `STUDY_CONFIG` environment variable is used to specify the name of the study or program that is being run inside the
Docker container. This variable is typically used by the application running inside the container to differentiate
between different studies or programs.

Note that the `STUDY_CONFIG` variable can be set to any string value, and should be set to a unique value for each
separate study or program.

## User Permissions

The following document outlines the permissions that a user can have within the dashboard application. The permission
is specified as a key-value pair, where the key is the name of the permission and the value is either `true` or `false`.
If the value is `true`, the user has access to the corresponding feature or data, and if the value is `false`, the user 
does not have access. You need to place these key-value pairs within the `admin_dashboard` key in one of the configs of 
the emission.

These are all the permissions that you can specify:

### Overview Page
- `overview_users`: User can see the total number of users in the Overview page.
- `overview_active_users`: User can see the number of active users in the Overview page.
- `overview_trips`: User can see the number of trips in the Overview page.
- `overview_signup_trends`: User can view the signup trend graph in the Overview page.
- `overview_trips_trend`: User can view the trip trend graph in the Overview page.

### Data Page
- `data_uuids`: User can view the UUIDs data in the Data page.
- `data_trips`: User can view the trips data in the Data page.
- `data_demographics`: User can view the demographics data in the Data page.
- `data_trajectories`: User can view the trajectories data in the Data page.
- `data_trips_columns_exclude`: It used to specify a list of column names that should be excluded from the trips data
that is displayed on the Data page. It includes valid columns from the **Stage_analysis_timeseries** collection. Valid
columns are specified in the following sections.
- `data_uuids_columns_exclude`: It used to specify a list of column names that should be excluded from the uuids data
that is displayed on the Data page. It includes valid columns from the **Stage_uuids** collection. Valid columns are 
specified in the following sections.
- `data_demographics_columns_exclude`: It used to specify a list of column names that should be excluded from the demographics data
that is displayed on the Data page. 
- `data_trajectories_columns_exclude`: It used to specify a list of column names that should be excluded from the trajectories data
that is displayed on the Data page.

### Token Page
- `token_generate`: User can generate new tokens in the Token page.
- `token_prefix`: The prefix that will be added to all tokens when creating new tokens.

### Map Page
- `map_heatmap`: User can view the heatmap in the Map page.
- `map_bubble`: User can view the bubble map in the Map page.
- `map_trip_lines`: User can view the trip lines map in the Map page.

### Segment Trip Time Page
- `segment_trip_time`: User can view this page. (default `true`)
- `segment_trip_time_full_trips`: User can see the table containing non-aggregated data (default `true`)
- `segment_trip_time_min_users`: Minimal number of distinct users in data required to display anything (value is a number, default `0`).

### Push Notification Page
- `push_send`: User can send push notifications in the Push Notification page.

### Dropdown Options
- `options_uuids`: User can see the UUIDs of users in dropdowns and choose between them.
- `options_emails`: User can see the emails of users in dropdowns and choose between them.
<br><br><br>


## Stage_uuids

This document represents a single row in a MongoDB collection. The document contains a single object of a uuid with the
following fields that you can provide in `data_uuids_columns_exlude`:

```python
valid_uuids_columns = [
    'user_token',
    'user_id',
    'update_ts',
]
```


## Stage_analysis_timeseries

It contains information about a segmentation trip. It has the following fields that you can provide in
`data_trips_columns_exclude`.

```python
valid_trip_columns = [
    "data.source",
    "data.start_ts",
    "data.start_local_dt",
    "data.start_fmt_tm",
    "data.start_place",
    "data.start_loc",
    "data.end_ts",
    "data.end_local_dt",
    "data.end_fmt_time",
    "data.end_place",
    "data.end_loc",
    "data.duration",
    "data.distance",
    "metadata.key",
    "metadata.platform",
    "metadata.write_ts",
    "metadata.time_zone",
    "metadata.write_local_dt",
    "metadata.write_fmt_time",
    "user_id",
]
```


# Authentication

## Specify Authentication Type

The AUTH_TYPE environment variable is used to specify the authentication type for the dashboard, and it is defined in
the Docker Compose file. It has two possible values: "basic" and "cognito". The "basic" option refers to basic
authentication, which is a simple way to authenticate users using a username and password. The "cognito" option refers
to Amazon Cognito, which is a user authentication service that can be used with AWS services.

## `config.py`

The `config.py` file is a Python module that contains configuration settings for an application that uses
authentication. To use this file, first make a copy of the provided `config-fake.py` file and rename it to `config.py`.
Then, fill in the necessary variables with your own valid data.

### CognitoConfig Class

The `CognitoConfig` class contains variables used for authentication when using AWS Cognito. To use this authentication
method, fill in the following variables with your credential data from your AWS Cognito panel in `config.py`:

- `CLIENT_ID`: This is a string that represents the client ID of the app that is registered with the user pool. When an
app wants to authenticate with a user pool, it must provide its client_id to the user pool's authentication server.
- `CLIENT_SECRET`: This is a string that represents the client secret of the app that is registered with the user pool.
The client_secret is a secret key that is used to authenticate the app with the user pool's authentication server. It
must be kept secure and not shared with anyone who should not have access to it.
- `REDIRECT_URL`: This is a string that represents the URL that users should be redirected to after they have
authenticated with the user pool's authentication server.
- `TOKEN_ENDPOINT`: This is a string that represents the endpoint for retrieving access tokens from the user pool's
authentication server. Access tokens are used by the app to access protected resources on behalf of the authenticated
user. It is your user pool's domain plus `/oauth2/token`.
- `USER_POOL_ID`: This is a string that represents the ID of the user pool that the app is registered with. The user
pool is a collection of users who can authenticate with the app.
- `REGION`: This is a string that represents the AWS region where the user pool is located. For example, "us-east-1" or
"eu-west-2".
- `AUTH_URL`: This is a string that represents the URL for initiating authentication requests with the user pool's
authentication server. It is the `Hosted UI` of your user pool.

### VALID_USERNAME_PASSWORD_PAIRS

The `VALID_USERNAME_PASSWORD_PAIRS` dictionary contains all the valid usernames and passwords that users can
authenticate with when using basic authentication. To use this authentication method, fill in the dictionary with your
own valid usernames and passwords in `config.py`.

### Usage

To use the configuration settings defined in `config.py`, import the module at the beginning of your Python script, and
access the variables using dot notation. For example:

```python
from config import CognitoConfig, VALID_USERNAME_PASSWORD_PAIRS

# Access the CLIENT_ID variable in CognitoConfig
client_id = CognitoConfig.CLIENT_ID

# Access the valid username and password pairs
valid_pairs = VALID_USERNAME_PASSWORD_PAIRS
