FROM python:3.9-slim

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN pip3 install --no-cache-dir --upgrade \
    pip \
    virtualenv

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common

USER appuser
WORKDIR /home/appuser

# Copy application files
COPY --chown=appuser:appuser . /home/appuser/app/

ENV VIRTUAL_ENV=/home/appuser/venv
RUN virtualenv ${VIRTUAL_ENV}
RUN . ${VIRTUAL_ENV}/bin/activate && pip install -r app/requirements.txt

# Set environment variables for Cloud Run
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Copy and set permissions for run script
COPY --chown=appuser:appuser run.sh /home/appuser/
RUN chmod +x /home/appuser/run.sh

# Create .streamlit directory
RUN mkdir -p /home/appuser/.streamlit

EXPOSE 8080

ENTRYPOINT ["./run.sh"]