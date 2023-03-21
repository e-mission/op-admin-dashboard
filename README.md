# Dockerized Dash App Template


https://towardsdatascience.com/dockerize-your-dash-app-1e155dd1cea3


Basic Dash app with data load, global variable module, and various UI examples.

### NREL Branding
This app uses the NREL Branding component, which is included as a .tgz and is installed via pip (see below).


## How to run this app

## Docker Compose method (recommended)

`docker compose -f docker-compose-dash-app.yml build`

`docker compose -f docker-compose-dash-app.yml up`


## Docker method

`docker build -t dash-app .`

`docker run dash-app`




## Windows Command line method

(The following instructions apply to Windows command line.)

Create and activate a new virtual environment (recommended) by running
the following:

On Windows

```
virtualenv venv
\venv\scripts\activate
```
On Mac
```
virtualenv venv
source venv/bin/activate
```

Or if using linux

```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

Install the requirements:

```
pip install -r dashboard/requirements.txt
```

Run the app:

```
python app.py
```
You can run the app on your browser at http://127.0.0.1:8050



## Resources

To learn more about Dash, please visit [documentation](https://plot.ly/dash).


## How to load data into the app

1. Run the app (steps shown above)
2. Copy your `.tar.gz` data file into this directory (`/op-admin-dashboard/<name of data file>.tar.gz`). If you do not have this file, speak with @shankari to get access to these files
3. Run the following shell script: `bash viz_scripts/docker/load_mongodump.sh vail_2023-01-01.tar-2.gz`. This will load the data into the admin dashboard

_**Note from Sebastian Barry, 3/10/2023**: This file does not exist on the `Dev` branch currently, so you will need to use the version of this file either from `sebastianbarry/op-admin-dashboard` branch named `readme-update`, or from the public dashboard: https://github.com/e-mission/em-public-dashboard/blob/main/viz_scripts/docker/load_mongodump.sh_

4. Open up the admin dashboard using your browser at http://127.0.0.1:8050 and you will be able to see the admin dashboard with loaded data