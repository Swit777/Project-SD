from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


SOURCE_DOI = "https://doi.org/10.17605/OSF.IO/WPEH6"
ARTICLE_URL = "https://www.nature.com/articles/s41597-023-02530-3"
FILES = {
    "data.zip": {
        "url": "https://osf.io/download/j48qf/?version=1",
        "bytes": 838310437,
        "sha256": "1b4d1f7daf61548d9f40b9b0b6ac9fe3e44ed0dff5d0d496fc002d08ac3aec41",
    },
    "codebook.xlsx": {
        "url": "https://osf.io/download/ye25m/?version=1",
        "bytes": 46383,
        "sha256": "c843e922b2978311967773fa330187369a3b7c86f34a01460a00d3ef9964b648",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sources(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"doi": SOURCE_DOI, "article": ARTICLE_URL, "license": "CC0-1.0", "files": {}}
    for name, spec in FILES.items():
        destination = directory / name
        if destination.exists():
            if destination.stat().st_size != spec["bytes"] or sha256_file(destination) != spec["sha256"]:
                raise ValueError(f"Existing file has the wrong checksum: {destination}")
            print(f"Verified cached source: {name}", flush=True)
        else:
            partial = directory / (name + ".part")
            for attempt in range(4):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == spec["bytes"]:
                    break
                headers = {"User-Agent": "PlayerExperienceResearch/0.1", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                try:
                    with urlopen(Request(spec["url"], headers=headers), timeout=60) as response:
                        resumed = offset > 0 and response.status == 206
                        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Server returned an incorrect resume range")
                        written = offset if resumed else 0
                        last_report = time.monotonic()
                        with partial.open("ab" if resumed else "wb") as output:
                            while block := response.read(1024 * 1024):
                                output.write(block)
                                written += len(block)
                                if time.monotonic() - last_report > 8:
                                    print(f"{name}: {written / spec['bytes']:.1%} ({written / 1e6:.1f} MB)", flush=True)
                                    last_report = time.monotonic()
                    if partial.stat().st_size == spec["bytes"]:
                        break
                except (OSError, TimeoutError) as error:
                    print(f"Download retry {attempt + 1}: {error}", flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(2)
            if not partial.exists() or partial.stat().st_size != spec["bytes"]:
                raise ValueError(f"Incomplete download: {name}")
            if sha256_file(partial) != spec["sha256"]:
                raise ValueError(f"Source checksum mismatch: {name}")
            partial.replace(destination)
            print(f"Downloaded and verified: {name}", flush=True)
        manifest["files"][name] = {**spec, "path": name}
    (directory / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
