import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "pipeline_job_store.py"
SPEC = importlib.util.spec_from_file_location("pipeline_job_store", SCRIPT_PATH)
pipeline_job_store = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pipeline_job_store
SPEC.loader.exec_module(pipeline_job_store)

connect = pipeline_job_store.connect
init_db = pipeline_job_store.init_db
add_manifest = pipeline_job_store.add_manifest
get_job = pipeline_job_store.get_job
list_jobs = pipeline_job_store.list_jobs
update_job = pipeline_job_store.update_job


def write_manifest(path):
    data = {
        "model_name": "test-model",
        "steps": [
            {"name": "ablate", "kind": "measurement", "gpu": True, "command": ["python", "ablate.py"], "outputs": ["a.json"]},
            {"name": "analyze", "kind": "analysis", "gpu": False, "command": ["python", "analyze.py"], "outputs": ["b.json"]},
        ],
    }
    path.write_text(json.dumps(data))
    return path


def test_add_manifest_creates_job_phases_and_event(tmp_path):
    conn = connect(tmp_path / "jobs.sqlite3")
    init_db(conn)
    manifest = write_manifest(tmp_path / "manifest.json")

    job_id = add_manifest(conn, manifest, priority=5)
    job = get_job(conn, job_id)

    assert job["job"]["model_name"] == "test-model"
    assert job["job"]["status"] == "queued"
    assert job["job"]["current_phase"] == "ablate"
    assert len(job["phases"]) == 2
    assert job["phases"][0]["gpu"] == 1
    assert job["events"][0]["event_type"] == "created"


def test_list_jobs_orders_by_priority(tmp_path):
    conn = connect(tmp_path / "jobs.sqlite3")
    init_db(conn)
    first = write_manifest(tmp_path / "first.json")
    second = write_manifest(tmp_path / "second.json")

    add_manifest(conn, first, priority=50)
    add_manifest(conn, second, priority=10)
    rows = list_jobs(conn)

    assert [row["priority"] for row in rows] == [10, 50]
    assert rows[0]["phase_count"] == 2
    assert rows[0]["gpu_phase_count"] == 1


def test_update_job_records_status_event(tmp_path):
    conn = connect(tmp_path / "jobs.sqlite3")
    init_db(conn)
    job_id = add_manifest(conn, write_manifest(tmp_path / "manifest.json"), priority=100)

    update_job(conn, job_id, status="running", phase="ablate", progress=0.25)
    job = get_job(conn, job_id)

    assert job["job"]["status"] == "running"
    assert job["job"]["progress"] == 0.25
    assert job["events"][-1]["event_type"] == "status"
