FROM apache/airflow:2.9.1-python3.11

# Install project dependencies as the airflow user (image default)
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy full project into the container
COPY . /opt/airflow/project
