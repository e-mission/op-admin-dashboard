#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <dump_file> <database_name> <docker_container_name> <collection_name>"
    exit 1
fi

# Assign arguments to variables
MONGODUMP_FILE="$1"
DATABASE_NAME="$2"
DOCKER_CONTAINER_NAME="$3"
COLLECTION_NAME="$4"

# Extract the base name of the dump file for use in the restore command
FILE_NAME=$(basename "$MONGODUMP_FILE")

# Drop the existing database to ensure it’s clean before restoring the new dump
echo "Dropping the existing database $DATABASE_NAME"
docker exec "$DOCKER_CONTAINER_NAME" mongo "$DATABASE_NAME" --eval "db.dropDatabase()"

# Check if the drop command was successful
if [ $? -ne 0 ]; then
    echo "Failed to drop the database $DATABASE_NAME"
    exit 1
fi

# Copy the MongoDB dump file from the local machine to the Docker container
echo "Copying file to Docker container $DOCKER_CONTAINER_NAME"
docker cp "$MONGODUMP_FILE" "$DOCKER_CONTAINER_NAME:/tmp"

# Check if the copy command was successful
if [ $? -ne 0 ]; then
    echo "Failed to copy the dump file to the Docker container"
    exit 1
fi

# Restore the dump into the specified database
echo "Restoring the dump from $FILE_NAME to database $DATABASE_NAME"
docker exec -e MONGODUMP_FILE="/tmp/$FILE_NAME" "$DOCKER_CONTAINER_NAME" bash -c \
    'tar xvf "$MONGODUMP_FILE" -C /tmp && mongorestore --db '"$DATABASE_NAME"' /tmp/dump/'"$COLLECTION_NAME"' --drop'

# Check if the restore command was successful
if [ $? -ne 0 ]; then
    echo "Failed to restore the dump to the database $DATABASE_NAME"
    exit 1
fi

echo "Restore completed successfully"
