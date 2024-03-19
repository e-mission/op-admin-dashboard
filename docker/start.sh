#!/bin/bash
source setup/activate.sh

# change the db host
echo "DB host = "${DB_HOST}
if [ -z ${DB_HOST} ] ; then
    local_host=`hostname -i`
    sed "s_localhost_${local_host}_" conf/storage/db.conf.sample > conf/storage/db.conf
else
    sed "s-localhost-${DB_HOST}-" conf/storage/db.conf.sample > conf/storage/db.conf
fi

# run the app
# python app.py
python app_sidebar_collapsible.py
