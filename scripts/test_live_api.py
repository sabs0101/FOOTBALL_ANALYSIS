import urllib.request
import json
import time

payload = {
    "source": "data/videos/sample_broadcast.mp4",
    "radar": True,
    "speed": True,
    "tactics": True,
    "heatmaps": True,
    "cmc": True,
    "reid": True,
    "events": True,
    "clahe": True,
}

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/process",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read().decode("utf-8"))
    task_id = data["task_id"]
    print("Started Task ID:", task_id)

    for i in range(120):
        time.sleep(1)
        p_req = urllib.request.urlopen(f"http://127.0.0.1:8000/api/progress?task_id={task_id}")
        p_data = json.loads(p_req.read().decode("utf-8"))
        status = p_data.get("status")
        cur_frame = p_data.get("current_frame")
        tot_frame = p_data.get("total_frames")
        fps = p_data.get("fps")
        err = p_data.get("error")
        print(f"[{i}s] Status: {status} | Frame: {cur_frame}/{tot_frame} | FPS: {fps} | Error: {err}")
        if status in ["completed", "error"]:
            print("Final Response:", json.dumps(p_data, indent=2))
            break
except Exception as e:
    print("API Error:", e)
