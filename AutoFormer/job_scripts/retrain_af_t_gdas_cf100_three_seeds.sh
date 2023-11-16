#!/bin/bash

# Path to the parameterized script
PARAMETERIZED_SCRIPT="job_scripts/retrain_job_submitter.sh"

# Run the parameterized script with the specified options
bash ${PARAMETERIZED_SCRIPT} --optimizer GDAS --dataset CF100 --array-task-range 9011,9021,9031
