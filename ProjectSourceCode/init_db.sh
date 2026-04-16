#!/bin/bash
# DO NOT PUSH THIS FILE TO GITHUB
# This file contains sensitive information and should be kept private
# TODO: Set your PostgreSQL URI - Use the External Database URL from the Render dashboard
PG_URI="postgresql://users_db_6xyo_user:2GMG9vLGtKZ2uxWlDA1bjoDJVuXrlCt7@dpg-d7girf471suc739f0qdg-a.oregon-postgres.render.com/users_db_6xyo"
# Execute each .sql file in the directory
for file in init_data/*.sql; do
    echo "Executing $file..."
    psql $PG_URI -f "$file"
done