FROM continuumio/miniconda3:latest
RUN conda create -n fake-news-env python=3.12
SHELL ["conda", "run", "-n", "fake-news-env", "/bin/bash", "-c"]
RUN AIRFLOW_VERSION=2.9.0 && \
    PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" && \
    CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" && \
    pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
WORKDIR /project
COPY . /project
CMD ["python", "app.py"]
