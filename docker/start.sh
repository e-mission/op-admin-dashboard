#!/bin/bash
source setup/activate.sh

# change the db host
echo ${DB_HOST}

# run the app
# python app.py
python app_sidebar_collapsible.py
