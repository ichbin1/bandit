"""HTTP-слой веб-интерфейса: схема, предпросмотр, задания, прогоны, файлы."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from bandit.web.app import _run_dir, create_app  # noqa: E402

TERMINAL = {"done", "failed", "cancelled"}


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as test_client:
        yield test_client


def _spec(**overrides):
    spec = {
        "title": "тест",
        "env": {"kind": "layered", "params": {"n_levels": 1, "max_arms": 5}},
        "agents": [{"key": "ucb1"}],
        "policies": [{"key": "none"}, {"key": "full_cycle"}],
        "runs": 1,
        "seed": 3,
        "steps": 40,
        "jobs": 1,
        "plots": False,
        "add_oracle": True,
    }
    spec.update(overrides)
    return spec


def _wait_job(client, job_id, timeout=120.0):
    deadline = time.time() + timeout
    job = client.get(f"/api/jobs/{job_id}").json()
    while time.time() < deadline and job["state"] not in TERMINAL:
        time.sleep(0.2)
        job = client.get(f"/api/jobs/{job_id}").json()
    return job


def test_health_and_schema(client):
    health = client.get("/api/health").json()
    assert health["version"]
    assert health["jobs_running"] == 0
    assert health["steps_per_second"] > 0
    payload = client.get("/api/schema").json()
    assert {env["kind"] for env in payload["environments"]} == {
        "layered",
        "pyramid",
        "bernoulli",
        "gaussian",
    }
    assert payload["defaults"]["steps"] > 0


def test_preview_and_validate(client):
    preview = client.post("/api/preview", json=_spec()).json()
    assert preview["n_configs"] == 3
    assert preview["n_episodes"] == 3
    assert preview["total_steps"] == 3 * 40
    assert preview["est_seconds"] >= 0
    assert "ui-layered-oracle-full_cycle" in preview["configs"]

    validate = client.post("/api/validate", json=_spec()).json()
    assert validate == {"ok": True, "n_configs": 3, "errors": []}


def test_unknown_field_is_rejected_with_422(client):
    payload = _spec()
    payload["unknown_option"] = 1
    response = client.post("/api/preview", json=payload)
    assert response.status_code == 422
    assert any("unknown_option" in str(item["loc"]) for item in response.json()["detail"])


def test_unknown_keys_and_bad_params_give_400(client):
    response = client.post("/api/preview", json=_spec(agents=[{"key": "nope"}]))
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]

    response = client.post("/api/jobs", json=_spec(env={"kind": "nope", "params": {}}))
    assert response.status_code == 422  # среда проверяется моделью

    bad_env = _spec(
        env={"kind": "bernoulli", "params": {"n_arms": 3, "means": [0.2]}},
        policies=[{"key": "none"}],
        add_oracle=False,
    )
    response = client.post("/api/jobs", json=bad_env)
    assert response.status_code == 400
    assert "means" in response.json()["detail"]


def test_missing_job_and_run_are_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/file/metrics.csv").status_code == 404
    assert client.post("/api/runs/nope/report").status_code == 404


def test_run_dir_rejects_escapes(tmp_path):
    with pytest.raises(HTTPException):
        _run_dir(tmp_path, "../etc")
    with pytest.raises(HTTPException):
        _run_dir(tmp_path, "with space")
    with pytest.raises(HTTPException):
        _run_dir(tmp_path, "absent")

def test_job_runs_cli_and_exposes_metrics(client, tmp_path):
    response = client.post("/api/jobs", json=_spec())
    assert response.status_code == 200
    job = response.json()
    assert job["n_configs"] == 3
    assert (tmp_path / job["id"] / "configs.json").is_file()
    assert (tmp_path / job["id"] / "spec.json").is_file()

    job = _wait_job(client, job["id"])
    assert job["state"] == "done", job["log"][-5:]
    assert job["n_done"] == 3
    assert job["progress"] == 1.0
    assert any("старт" in line for line in job["log"])

    assert [item["id"] for item in client.get("/api/jobs").json()] == [job["id"]]

    run = client.get(f"/api/runs/{job['id']}").json()
    assert set(run["configs"]) == {
        "ui-layered-ucb1-none",
        "ui-layered-ucb1-full_cycle",
        "ui-layered-oracle-full_cycle",
    }
    assert run["n_runs"] == 1
    assert run["n_steps"] == 40
    assert run["has_report"] is False
    assert run["has_plots"] is False
    row = next(item for item in run["metrics"] if item["config"] == "ui-layered-ucb1-none")
    assert row["values"]["final_points"]["mean"] is not None

    listing = client.get("/api/runs").json()
    assert listing and listing[0]["id"] == job["id"]

    csv_response = client.get(f"/api/runs/{job['id']}/file/metrics.csv")
    assert csv_response.status_code == 200
    assert "final_points" in csv_response.text

    assert client.get(f"/api/runs/{job['id']}/file/secret.txt").status_code == 404


def test_job_can_be_cancelled(client):
    job = client.post("/api/jobs", json=_spec(steps=200_000)).json()
    cancelled = client.post(f"/api/jobs/{job['id']}/cancel").json()
    assert cancelled["state"] in {"queued", "running", "cancelled"}
    job = _wait_job(client, job["id"], timeout=60.0)
    assert job["state"] == "cancelled", job["log"][-5:]


def test_report_endpoint_writes_artifacts(client, tmp_path):
    job = client.post("/api/jobs", json=_spec()).json()
    job = _wait_job(client, job["id"])
    assert job["state"] == "done", job["log"][-5:]

    response = client.post(f"/api/runs/{job['id']}/report")
    assert response.status_code == 200, response.text
    run = response.json()
    assert run["has_report"] is True
    assert (tmp_path / job["id"] / "report.md").is_file()
    assert (tmp_path / job["id"] / "report.json").is_file()
    assert client.get(f"/api/runs/{job['id']}/file/report.md").status_code == 200


def test_static_index_is_served_last(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "bandit" in index.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_module_entry_point_passes_arguments(monkeypatch):
    from bandit.web import __main__ as web_main

    calls = {}

    def fake_serve(**kwargs):
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(web_main, "serve", fake_serve)
    assert web_main.main(["--port", "8321", "--no-browser"]) == 0
    assert calls == {"host": "127.0.0.1", "port": 8321, "open_browser": False}
    assert web_main.main([]) == 0
    assert calls == {"host": "127.0.0.1", "port": 8000, "open_browser": True}


