"""
dags/lgbtiq_compliance_dag.py
------------------------------
Monthly DAG for the LGBTIQ+ rights compliance pipeline.

Task order:
    extract_all_states >> [transform_{state} × 32] >> load_results

Data flow:
    extract_all_states  — scrapes / reads PDFs for all 32 states sequentially,
                          extracts text with pdfplumber, writes .txt files to
                          data/raw/; returns {state: [file_path, ...]} via XCom
    transform_{state}   — one task per state (pool=transform_pool, 3 slots);
                          builds FAISS vector store, runs LangGraph devil's
                          advocate graph for 6 markers; returns list of 6 records
    load_results        — merges XCom from all 32 transform tasks, writes JSON
                          to data/output/ and upserts to Atlas
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from tasks.extract import STATES, extract_all_states
from tasks.transform import transform_state
from tasks.load import load_results

with DAG(
    dag_id="lgbtiq_compliance_pipeline",
    description=(
        "Monthly legal gap analysis of Mexican state laws for LGBTIQ+ "
        "rights compliance (markers L1–L3, C1–C3) using pdfplumber + LangGraph."
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

    # ── Extract (all states, sequential) ─────────────────────────────────────
    t_extract = PythonOperator(
        task_id="extract_all_states",
        python_callable=extract_all_states,
        execution_timeout=timedelta(hours=3),
        pool="extract_pool",
    )

    # ── Transform (one task per state, max 3 concurrent via pool) ────────────
    transform_tasks = []
    for state in STATES:
        task_id = f"transform_{state.replace(' ', '_')}"
        t = PythonOperator(
            task_id=task_id,
            python_callable=transform_state,
            op_kwargs={"state_name": state},
            execution_timeout=timedelta(minutes=30),
            pool="transform_pool",
        )
        transform_tasks.append(t)

    # ── Load (consolidates all transform XCom results) ────────────────────────
    t_load = PythonOperator(
        task_id="load_results",
        python_callable=load_results,
        execution_timeout=timedelta(minutes=30),
    )

    t_extract >> transform_tasks >> t_load
