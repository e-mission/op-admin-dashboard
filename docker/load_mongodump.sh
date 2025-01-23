#!/bin/bash

# Directory of the script
SCRIPT_DIR="$(dirname "$0")"

# Path to the configuration file (one level up)
CONFIG_FILE="$SCRIPT_DIR/../docker-compose-dev.yml"

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <mongodump-file>"
    echo "  <mongodump-file> : The path to the MongoDB dump file to be restored."
    exit 1
fi

MONGODUMP_FILE=$1

# Print debug information
echo "Script Directory: $SCRIPT_DIR"
echo "Configuration File Path: $CONFIG_FILE"
echo "MongoDump File Path: $MONGODUMP_FILE"

# Check if the provided file exists
if [ ! -f "$MONGODUMP_FILE" ]; then
    echo "Error: File '$MONGODUMP_FILE' does not exist."
    exit 1
fi

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' does not exist."
    exit 1
fi

# Print details about the configuration file
echo "Configuration file details:"
ls -l "$CONFIG_FILE"

# Extract the database name from the mongodump file
DB_NAME=$(tar -tf "$MONGODUMP_FILE" | grep '^dump/' | sed 's|^dump/||' | awk -F'/' '{if (NF > 0) {print $1; exit}}')

# Output the database name
echo "$DB_NAME"

if [ -z "$DB_NAME" ]; then
    echo "Error: Failed to extract database name from mongodump."
    exit 1
fi

echo "Database Name: $DB_NAME"

# Update the docker-compose configuration file with the actual DB_HOST
DB_HOST="mongodb://db/$DB_NAME"
sed -i.bak "s|DB_HOST:.*|DB_HOST: $DB_HOST|" "$CONFIG_FILE"

# Update the docker-compose configuration file with the actual STUDY_CONFIG
STUDY_CONFIG=$(echo "$DB_NAME" | sed -E 's/openpath_prod_(.*)$/\1/' | tr '_' '-')
sed -i.bak "s|STUDY_CONFIG:.*|STUDY_CONFIG: \"$STUDY_CONFIG\"|" "$CONFIG_FILE"

echo "Updated docker-compose file:"
cat "$CONFIG_FILE"

echo "Copying file to Docker container"
docker cp "$MONGODUMP_FILE" op-admin-dashboard-db-1:/tmp

FILE_NAME=$(basename "$MONGODUMP_FILE")

echo "Clearing existing database"
docker exec op-admin-dashboard-db-1 bash -c "mongo $DB_NAME --eval 'db.dropDatabase()'"

echo "Restoring the dump from $FILE_NAME to database $DB_NAME"
docker exec -e MONGODUMP_FILE=$FILE_NAME op-admin-dashboard-db-1 bash -c "cd /tmp && tar xvf $FILE_NAME && mongorestore -d $DB_NAME dump/$DB_NAME"

echo "Database restore complete."