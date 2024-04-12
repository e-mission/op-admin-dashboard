#!/bin/bash
source setup/activate.sh

# change the db host
echo "DB host = "${DB_HOST}
if [ -z ${DB_HOST} ] ; then
    local_host=`hostname -i`
    export DB_HOST=$local_host
    echo "Setting db host environment variable to localhost"
fi

# run the app
# python app.py
python app_sidebar_collapsible.py
