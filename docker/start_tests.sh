#!/bin/bash
source setup/activate.sh

# default to tests/ since the working dir also contains the e-mission server tests
pytest "${@:-tests}"
