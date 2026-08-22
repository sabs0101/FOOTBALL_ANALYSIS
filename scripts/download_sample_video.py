"""
Download Bundesliga sample broadcast soccer match video in mp4.
"""

from pathlib import Path
import re
import requests
from tqdm import tqdm


def download_file_from_google_drive(file_id: str, destination: Path):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    # Check if download warning exists in HTML response
    match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
    if match:
        return match.group(1)
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        desc=destination.name,
        total=total_size if total_size > 0 else None,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


if __name__ == "__main__":
    dest = Path("data/videos/sample_broadcast.mp4")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Bundesliga broadcast clip (08fd33_0.mp4) to {dest}...")
    download_file_from_google_drive("1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-", dest)
    print(f"Done. File size: {dest.stat().st_size / (1024*1024):.2f} MB")
