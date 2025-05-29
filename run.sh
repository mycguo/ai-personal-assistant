#!/bin/bash
set -e

echo "Starting application..."

APP_PID=
stopRunningProcess() {
    echo "Received shutdown signal..."
    if test ! "${APP_PID}" = '' && ps -p ${APP_PID} > /dev/null ; then
        echo "Stopping process with ID ${APP_PID}"
        kill -TERM ${APP_PID}
        echo "Waiting for process to handle SIGTERM"
        wait ${APP_PID}
        echo "Process stopped"
    else
        echo "Process was not running or already stopped"
    fi
}

trap stopRunningProcess EXIT TERM

echo "Activating virtual environment..."
source ${VIRTUAL_ENV}/bin/activate

echo "Starting Streamlit on port ${PORT}..."
streamlit run ${HOME}/app/app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.serverAddress=0.0.0.0 \
    --browser.gatherUsageStats=false &
APP_PID=${!}

echo "Streamlit started with PID ${APP_PID}"
wait ${APP_PID}