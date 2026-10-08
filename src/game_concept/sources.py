from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

DATASET_ID = "FronkonGames/steam-games-dataset"
REVISION = "ba4e26785af33bee500e96597068c6e77f4edea9"
SNAPSHOT_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/games.csv"
SNAPSHOT_BYTES = 401140503
SNAPSHOT_SHA256 = "1b48008b01a799d82385d6e66ba0cb65dd477b6aa4b6c2ec8c443025ab6880ad"
USER_AGENT = "GameConceptResearch/0.1 (academic public-metadata analysis; cached; no login)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def download_snapshot(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "games.csv"
    if destination.exists():
        if destination.stat().st_size != SNAPSHOT_BYTES or sha256_file(destination) != SNAPSHOT_SHA256:
            raise ValueError("Snapshot checksum mismatch; preserve the altered file and restore the pinned source")
        print("Verified pinned Steam snapshot", flush=True)
    else:
        part = directory / "games.csv.part"
        for attempt in range(4):
            offset = part.stat().st_size if part.exists() else 0
            if offset == SNAPSHOT_BYTES:
                break
            headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with requests.get(SNAPSHOT_URL, headers=headers, stream=True, timeout=(15, 90)) as response:
                    response.raise_for_status()
                    resumed = offset > 0 and response.status_code == 206
                    if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Incorrect download resume range")
                    written = offset if resumed else 0
                    last_report = time.monotonic()
                    with part.open("ab" if resumed else "wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
                            written += len(block)
                            if written > SNAPSHOT_BYTES:
                                raise ValueError("Snapshot exceeds the pinned size")
                            if time.monotonic() - last_report > 8:
                                print(f"Steam snapshot: {written / SNAPSHOT_BYTES:.1%}", flush=True)
                                last_report = time.monotonic()
                if part.stat().st_size == SNAPSHOT_BYTES:
                    break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        if part.stat().st_size != SNAPSHOT_BYTES or sha256_file(part) != SNAPSHOT_SHA256:
            raise ValueError("Downloaded snapshot fails size/SHA-256 verification")
        part.replace(destination)
    manifest = {"dataset": DATASET_ID, "revision": REVISION, "url": SNAPSHOT_URL,
                "bytes": SNAPSHOT_BYTES, "sha256": SNAPSHOT_SHA256, "declared_dataset_license": "MIT",
                "source_file_updated_utc": "2026-06-20", "download_verified_utc": utc_now(),
                "rights_note": "Compilation license does not transfer third-party descriptions, artwork or trademarks. Raw data stays local."}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


class CachedHttp:
    """Sequential, bounded, checksum-verified access to public Steam metadata."""

    def __init__(self, directory: Path, interval: float = 2.0, refresh: bool = False):
        if interval < 1.5:
            raise ValueError("Request interval must be >= 1.5 seconds")
        self.directory, self.interval, self.refresh = directory, interval, refresh
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
        self.robot = None

    def get(self, url: str) -> tuple[bytes, dict]:
        parsed = urlparse(url)
        host = parsed.hostname
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise ValueError("Only public HTTPS endpoints without credentials are allowed")
        if host not in {"store.steampowered.com", "api.steampowered.com", "steamspy.com"}:
            raise ValueError("Only approved public Steam data hosts are allowed")
        if host == "store.steampowered.com" and self.robot is not None and not self.robot.can_fetch(USER_AGENT, url):
            raise ValueError("URL is disallowed by Steam robots.txt")
        key = hashlib.sha256(url.encode()).hexdigest()
        path, meta_path = self.directory / f"{key}.body", self.directory / f"{key}.json"
        if path.exists() and meta_path.exists() and not self.refresh:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata["sha256"] != sha256_file(path):
                raise ValueError("HTTP cache checksum mismatch")
            return path.read_bytes(), metadata
        for attempt in range(3):
            time.sleep(max(self.interval - (time.monotonic() - self.last_request), 0))
            self.last_request = time.monotonic()
            response = self.session.get(url, timeout=(15, 45))
            if response.status_code in (401, 403):
                raise PermissionError(f"Steam access denied ({response.status_code}); no bypass attempted")
            if response.status_code == 429:
                if attempt == 2:
                    response.raise_for_status()
                try:
                    delay = float(response.headers.get("Retry-After", "20"))
                except ValueError:
                    delay = 20
                if delay > 120:
                    raise RuntimeError("Steam requests a long pause; resume collection later")
                time.sleep(max(delay, self.interval))
                continue
            response.raise_for_status()
            body = response.content
            metadata = {"url": url, "captured_at_utc": utc_now(), "status": response.status_code,
                        "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
            archive = self.directory / "snapshots"
            archive.mkdir(exist_ok=True)
            if path.exists() and meta_path.exists():
                old = json.loads(meta_path.read_text(encoding="utf-8"))
                if sha256_file(path) != old["sha256"]:
                    raise ValueError("Cannot refresh a damaged HTTP cache")
                (archive / f"{old['sha256']}.body").write_bytes(path.read_bytes())
                (archive / f"{key}-{old['sha256']}.json").write_text(json.dumps(old, indent=2), encoding="utf-8")
            (archive / f"{metadata['sha256']}.body").write_bytes(body)
            (archive / f"{key}-{metadata['sha256']}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            path.write_bytes(body)
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return body, metadata
        raise RuntimeError("Request retry limit exceeded")

    def check_robots(self):
        body, _ = self.get("https://store.steampowered.com/robots.txt")
        self.robot = RobotFileParser()
        self.robot.parse(body.decode("utf-8").splitlines())
