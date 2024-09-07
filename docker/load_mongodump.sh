MONGODUMP_FILE=$1

echo "Copying file to docker container"
docker cp $MONGODUMP_FILE op-admin-dashboard-db-1:/tmp

FILE_NAME=`basename $MONGODUMP_FILE`

echo "Restoring the dump from $FILE_NAME"
docker exec -e MONGODUMP_FILE=$FILE_NAME op-admin-dashboard-db-1 bash -c 'cd /tmp && tar xvf $MONGODUMP_FILE && mongorestore'

