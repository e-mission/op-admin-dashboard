#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dump_file> <database_name> <docker_container_name>"
    exit 1
fi

# Assign arguments to variables
MONGODUMP_FILE=$1
DATABASE_NAME=$2
DOCKER_CONTAINER_NAME=$3

# Extract the base name of the dump file for use in the restore command
FILE_NAME=$(basename $MONGODUMP_FILE)

# Drop the existing database to ensure it’s clean before restoring the new dump
echo "Dropping the existing database $DATABASE_NAME"
docker exec $DOCKER_CONTAINER_NAME mongo $DATABASE_NAME --eval "db.dropDatabase()"

# Copy the MongoDB dump file from the local machine to the Docker container
echo "Copying file to Docker container $DOCKER_CONTAINER_NAME"
docker cp $MONGODUMP_FILE $DOCKER_CONTAINER_NAME:/tmp

# Restore the dump into the specified database
echo "Restoring the dump from $FILE_NAME to database $DATABASE_NAME"
docker exec -e MONGODUMP_FILE=$FILE_NAME $DOCKER_CONTAINER_NAME bash -c \
    'cd /tmp && tar xvf $MONGODUMP_FILE && mongorestore -d '"$DATABASE_NAME"' dump/openpath_prod_ca_ebike'
