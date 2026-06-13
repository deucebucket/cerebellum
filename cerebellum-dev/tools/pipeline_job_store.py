#!/usr/bin/env python3
"""SQLite job store for Cerebellum pipeline manifests."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_name TEXT NOT NULL,
  manifest_path TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  priority INTEGER NOT NULL DEFAULT 100,
  current_phase TEXT,
  progress REAL NOT NULL DEFAULT 0.0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_phases (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER NOT NULL,
  phase_index INTEGER NOT NULL,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,
  gpu INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued',
  command_json TEXT NOT NULL,
  outputs_json TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  error_message TEXT,
  FOREIGN KEY(job_id) REFERENCES pipeline_jobs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pipeline_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER NOT NULL,
  phase_id INTEGER,
  event_type TEXT NOT NULL,
  message TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(job_id) REFERENCES pipeline_jobs(id) ON DELETE CASCADE,
  FOREIGN KEY(phase_id) REFERENCES pipeline_phases(id) ON DELETE SET NULL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def add_manifest(conn: sqlite3.Connection, manifest_path: Path, priority: int) -> int:
    data = json.loads(manifest_path.read_text())
    steps = data.get("steps") or []
    model_name = data.get("model_name") or manifest_path.parent.name
    with conn:
        cur = conn.execute(
            """
            INSERT INTO pipeline_jobs (model_name, manifest_path, priority, current_phase, config_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (model_name, str(manifest_path), priority, steps[0]["name"] if steps else None, json.dumps(data)),
        )
        job_id = int(cur.lastrowid)
        for idx, step in enumerate(steps):
            conn.execute(
                """
                INSERT INTO pipeline_phases
                  (job_id, phase_index, name, kind, gpu, command_json, outputs_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    idx,
                    step["name"],
                    step.get("kind", "unknown"),
                    1 if step.get("gpu") else 0,
                    json.dumps(step.get("command") or []),
                    json.dumps(step.get("outputs") or []),
                ),
            )
        conn.execute(
            "INSERT INTO pipeline_events (job_id, event_type, message) VALUES (?, ?, ?)",
            (job_id, "created", f"queued manifest {manifest_path}"),
        )
    return job_id


def update_job(conn: sqlite3.Connection, job_id: int, status: str, phase: str | None, progress: float | None) -> None:
    fields = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
    values: list[Any] = [status]
    if phase is not None:
        fields.append("current_phase = ?")
        values.append(phase)
    if progress is not None:
        fields.append("progress = ?")
        values.append(progress)
    values.append(job_id)
    with conn:
        conn.execute(f"UPDATE pipeline_jobs SET {', '.join(fields)} WHERE id = ?", values)
        conn.execute(
            "INSERT INTO pipeline_events (job_id, event_type, message) VALUES (?, ?, ?)",
            (job_id, "status", f"status={status} phase={phase} progress={progress}"),
        )


def list_jobs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT j.*,
               COUNT(p.id) AS phase_count,
               SUM(CASE WHEN p.gpu THEN 1 ELSE 0 END) AS gpu_phase_count
        FROM pipeline_jobs j
        LEFT JOIN pipeline_phases p ON p.job_id = j.id
        GROUP BY j.id
        ORDER BY j.priority ASC, j.id ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def get_job(conn: sqlite3.Connection, job_id: int) -> dict[str, Any]:
    job = conn.execute("SELECT * FROM pipeline_jobs WHERE id = ?", (job_id,)).fetchone()
    if job is None:
        raise SystemExit(f"job {job_id} not found")
    phases = conn.execute(
        "SELECT * FROM pipeline_phases WHERE job_id = ? ORDER BY phase_index",
        (job_id,),
    ).fetchall()
    events = conn.execute(
        "SELECT * FROM pipeline_events WHERE job_id = ? ORDER BY id",
        (job_id,),
    ).fetchall()
    return {"job": dict(job), "phases": [dict(row) for row in phases], "events": [dict(row) for row in events]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("cerebellum-dev/pipeline_jobs.sqlite3"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    add = sub.add_parser("add-manifest")
    add.add_argument("manifest", type=Path)
    add.add_argument("--priority", type=int, default=100)
    sub.add_parser("list")
    get = sub.add_parser("get")
    get.add_argument("job_id", type=int)
    update = sub.add_parser("update")
    update.add_argument("job_id", type=int)
    update.add_argument("--status", required=True)
    update.add_argument("--phase")
    update.add_argument("--progress", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = connect(args.db)
    init_db(conn)
    if args.cmd == "init":
        print(f"initialized {args.db}")
    elif args.cmd == "add-manifest":
        job_id = add_manifest(conn, args.manifest, args.priority)
        print(json.dumps(get_job(conn, job_id), indent=2))
    elif args.cmd == "list":
        print(json.dumps({"jobs": list_jobs(conn)}, indent=2))
    elif args.cmd == "get":
        print(json.dumps(get_job(conn, args.job_id), indent=2))
    elif args.cmd == "update":
        update_job(conn, args.job_id, args.status, args.phase, args.progress)
        print(json.dumps(get_job(conn, args.job_id), indent=2))


if __name__ == "__main__":
    main()
