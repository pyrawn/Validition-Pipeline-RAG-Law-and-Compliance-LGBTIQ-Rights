"""
dags/lgbtiq_compliance_dag.py
------------------------------
Monthly DAG for the LGBTIQ+ rights compliance pipeline.

Task order:
    extract_pdfs >> transform_to_scores >> load_results

Data flow:
    extract_pdfs        — scrapes / reads PDFs, extracts text with Docling
                          returns dict {state: [(file_name, text), ...]}
    transform_to_scores — builds per-state vector stores, runs LangGraph
                          devil's advocate graph for 6 markers per state
                          returns list of 192 validated compliance records
    load_results        — writes JSON to data/output/ and upserts to Atlas
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from tasks.extract   import extract_pdfs
from tasks.transform import transform_to_scores
from tasks.load      import load_results

with DAG(
    dag_id="lgbtiq_compliance_pipeline",
    description=(
        "Monthly legal gap analysis of Mexican state laws for LGBTIQ+ "
        "rights compliance (markers L1–L3, C1–C3) using Docling + LangGraph."
    ),
    schedule_interval="@monthly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["lgbtiq", "compliance", "rag", "legal"],
    default_args={
        "owner":   "airflow",
        "retries": 1,
    },
) as dag:

    t_extract = PythonOperator(
        task_id="extract_pdfs",
        python_callable=extract_pdfs,
        execution_timeout=timedelta(hours=1),
    )

    t_transform = PythonOperator(
        task_id="transform_to_scores",
        python_callable=transform_to_scores,
        execution_timeout=timedelta(hours=2),
    )

    t_load = PythonOperator(
        task_id="load_results",
        python_callable=load_results,
        execution_timeout=timedelta(minutes=30),
    )

    t_extract >> t_transform >> t_load
