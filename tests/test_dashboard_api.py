"""
Unit Tests for Web Dashboard HTTP Server & REST API Endpoints (Milestone 12).
"""

import http.server
import json
import threading
import time
import urllib.request
import urllib.error
import pytest
from pathlib import Path

from app import DashboardHTTPRequestHandler, TASKS, TASKS_LOCK


@pytest.fixture(scope="module")
def server_port():
    """Spin up a test instance of the dashboard server on an ephemeral port."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), DashboardHTTPRequestHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)
    yield port
    server.shutdown()


def test_static_index_serving(server_port):
    url = f"http://127.0.0.1:{server_port}/"
    req = urllib.request.urlopen(url)
    assert req.status == 200
    assert "text/html" in req.headers.get("Content-Type", "")
    content = req.read().decode("utf-8")
    assert "AI Football Tactical Analytics" in content or "Match Tactical Intelligence" in content


def test_static_css_and_js_serving(server_port):
    css_url = f"http://127.0.0.1:{server_port}/styles.css"
    req_css = urllib.request.urlopen(css_url)
    assert req_css.status == 200
    assert "text/css" in req_css.headers.get("Content-Type", "")

    js_url = f"http://127.0.0.1:{server_port}/app.js"
    req_js = urllib.request.urlopen(js_url)
    assert req_js.status == 200
    assert "javascript" in req_js.headers.get("Content-Type", "")


def test_progress_endpoint_not_found(server_port):
    url = f"http://127.0.0.1:{server_port}/api/progress?task_id=fake_task_999"
    req = urllib.request.urlopen(url)
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert data["status"] == "not_found"


def test_progress_endpoint_with_active_task(server_port):
    with TASKS_LOCK:
        TASKS["test_task_123"] = {
            "status": "processing",
            "current_frame": 45,
            "total_frames": 100,
            "fps": 8.5,
            "player_count": 22,
            "results": None,
            "error": None,
        }

    url = f"http://127.0.0.1:{server_port}/api/progress?task_id=test_task_123"
    req = urllib.request.urlopen(url)
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert data["status"] == "processing"
    assert data["current_frame"] == 45
    assert data["fps"] == 8.5


def test_process_trigger_endpoint(server_port):
    url = f"http://127.0.0.1:{server_port}/api/process"
    payload = {
        "source": "data/videos/sample_broadcast.mp4",
        "radar": True,
        "speed": True,
        "tactics": True,
        "heatmaps": False,
        "cmc": True,
        "reid": True,
        "events": True,
        "clahe": True,
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req)
    assert resp.status == 200
    res_data = json.loads(resp.read().decode("utf-8"))
    assert res_data["status"] == "started"
    assert "task_id" in res_data
    task_id = res_data["task_id"]

    # Verify task was registered in memory
    with TASKS_LOCK:
        assert task_id in TASKS


def test_video_byte_range_request(server_port):
    test_video_path = Path("data/videos/sample_broadcast.mp4")
    if not test_video_path.exists():
        pytest.skip("Sample broadcast video not present for range testing")

    url = f"http://127.0.0.1:{server_port}/data/videos/sample_broadcast.mp4"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1023"})
    resp = urllib.request.urlopen(req)
    assert "bytes 0-1023/" in resp.headers.get("Content-Range", "")
    data = resp.read()
    assert len(data) == 1024


def test_video_upload_endpoint(server_port, tmp_path):
    url = f"http://127.0.0.1:{server_port}/api/upload"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    filename = "test_custom_match.mp4"
    dummy_video_content = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42" + b"A" * 1024

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="video"; filename="{filename}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode("utf-8") + dummy_video_content + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    resp = urllib.request.urlopen(req)
    assert resp.status == 200
    res_data = json.loads(resp.read().decode("utf-8"))
    assert res_data["status"] == "success"
    assert "filepath" in res_data
    uploaded_file = Path(res_data["filepath"])
    assert uploaded_file.exists()
    assert uploaded_file.stat().st_size == len(dummy_video_content)

