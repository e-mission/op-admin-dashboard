MONGODUMP_FILE=$1

echo "Copying file to docker container"
docker cp $MONGODUMP_FILE em-public-dashboard_db_1:/tmp

FILE_NAME=`basename $MONGODUMP_FILE`

echo "Restoring the dump from $FILE_NAME"
docker exec -e MONGODUMP_FILE=$FILE_NAME em-public-dashboard_db_1 bash -c 'cd /tmp && tar xvf $MONGODUMP_FILE && mongorestore && rm $MONGODUMP_FILE'
