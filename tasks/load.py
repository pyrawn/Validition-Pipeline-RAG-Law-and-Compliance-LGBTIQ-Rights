"""
tasks/load.py
-------------
Load task for the LGBTIQ+ compliance pipeline.

Receives validated compliance records from the transform task via XCom
and persists them in two destinations:

  1. Local JSON — data/output/scores_{execution_date}.json
  2. MongoDB Atlas — upsert per (state, marker) composite key

MongoDB credentials are read from config/mongo.yaml.
If MongoDB fails, the error is logged and the task still succeeds
(local JSON is the authoritative fallback).
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

from tasks.extract import STATES

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
CONFIG_PATH  = PROJECT_ROOT / "config" / "mongo.yaml"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "output"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_mongo_config() -> Dict[str, Any]:
    """
    Read config/mongo.yaml and return the 'mongo' section as a dict.
    Raises FileNotFoundError if the config file is missing.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"MongoDB config not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw["mongo"]


def _build_mongo_uri(cfg: Dict[str, Any]) -> str:
    """
    Return the Atlas SRV connection string from the YAML config.
    The 'host' field already contains the full mongodb+srv:// URI.
    """
    return cfg["host"]


def _save_json(records: List[Dict], execution_date: str) -> pathlib.Path:
    """
    Save all records to data/output/scores_{execution_date}.json.
    Returns the path written.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"scores_{execution_date}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return out_path


def _upsert_mongo(records: List[Dict], cfg: Dict[str, Any]) -> int:
    """
    Upsert all records into the configured MongoDB Atlas collection.

    Uses a bulk_write with UpdateOne operations and upsert=True so that
    re-running the pipeline overwrites previous scores for the same
    (state, marker) pair rather than creating duplicates.

    Returns the number of records upserted/modified.
    """
    uri        = _build_mongo_uri(cfg)
    db_name    = cfg["database"]
    col_name   = cfg["collection"]

    client     = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    collection = client[db_name][col_name]

    operations = [
        UpdateOne(
            filter={"state": r["state"], "marker": r["marker"]},
            update={"$set": r},
            upsert=True,
        )
        for r in records
    ]

    result = collection.bulk_write(operations, ordered=False)
    client.close()

    touched = result.upserted_count + result.modified_count
    return touched


# ── Airflow callable ──────────────────────────────────────────────────────────

def load_results(ti) -> None:
    """
    Airflow task callable — entry point for the load step.

    Pulls the list of validated compliance records from XCom
    (task_id='transform_to_scores') and writes them to:

    1. data/output/scores_{execution_date}.json  (always)
    2. MongoDB Atlas collection (best-effort — failure is logged, not raised)

    Logs the count of records written to each destination.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    # Pull per-state lists from all transform tasks and flatten into one list.
    transform_task_ids = [f"transform_{s.replace(' ', '_')}" for s in STATES]
    xcom_results = ti.xcom_pull(task_ids=transform_task_ids)

    records: List[Dict] = []
    for result in (xcom_results or []):
        if result:
            records.extend(result)

    if not records:
        raise ValueError("XCom returned empty records from all transform tasks.")

    execution_date = ti.execution_date.strftime("%Y-%m-%d")
    log.info("Loading %d records for execution_date=%s", len(records), execution_date)

    # ── Load 1: Local JSON ────────────────────────────────────────────────────
    json_path = _save_json(records, execution_date)
    log.info("JSON written: %s  (%d records)", json_path, len(records))

    # ── Load 2: MongoDB Atlas ─────────────────────────────────────────────────
    try:
        cfg     = _load_mongo_config()
        touched = _upsert_mongo(records, cfg)
        log.info("MongoDB upsert complete: %d records touched.", touched)
    except FileNotFoundError as exc:
        log.error("MongoDB config missing — skipping Atlas load. %s", exc)
    except PyMongoError as exc:
        log.error(
            "MongoDB write failed — local JSON is the fallback. Error: %s", exc
        )
    except Exception as exc:
        log.error("Unexpected error during MongoDB load: %s", exc)
