#!/bin/bash
source setup/activate.sh

# change the db host
echo "DB host = "${DB_HOST}
if [ -z ${DB_HOST} ] ; then
    local_host=`hostname -i`
    jq --arg db_host "$local_host" '.timeseries.url = $db_host' conf/storage/db.conf.sample > conf/storage/db.conf
else
    jq --arg db_host "$DB_HOST" '.timeseries.url = $db_host' conf/storage/db.conf.sample > conf/storage/db.conf
fi

# run the app
# python app.py
python app_sidebar_collapsible.py
