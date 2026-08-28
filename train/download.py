#!/usr/bin/env python3
"""Download declared training sources with explicit research-only opt-in."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((Path(__file__).with_name("sources.json")).read_text())
PROTOTYPE = {
    "Clean.zip",
    "BluesDriver.zip",
    "Digital-Delay.zip",
    "Hall-Reverb.zip",
    "Chorus.zip",
    "P1_singlenotes.zip",
    "P2_singlenotes.zip",
    "P3_music.zip",
    "audio_mono-mic.zip",
    "audio_mono-pickup_mix.zip",
}
EXTENDED_TECHS = {
    "P1_chords.zip",
    "P1_scales.zip",
    "P1_techniques.zip",
    "P2_chords.zip",
    "P2_scales.zip",
    "P2_techniques.zip",
}


def digest(path: Path) -> str:
    value = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def fetch(url: str, output: Path, size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        offset = output.stat().st_size if output.exists() else 0
        if offset == size:
            return
        if offset > size:
            output.unlink()
            offset = 0
        request = urllib.request.Request(url)
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                content_range = response.headers.get("Content-Range", "")
                resumes = (
                    offset
                    and response.status == 206
                    and content_range.startswith(f"bytes {offset}-")
                )
                mode = "ab" if resumes else "wb"
                with output.open(mode) as target:
                    shutil.copyfileobj(response, target, length=1024 * 1024)
            if output.stat().st_size == size:
                return
        except (OSError, urllib.error.URLError) as error:
            if attempt == 5:
                raise RuntimeError(f"could not download {url}: {error}") from error
        time.sleep(2**attempt)
    raise RuntimeError(f"download has the wrong size: {output}")


def extract(archive: Path, output: Path) -> None:
    marker = output / ".complete"
    if marker.exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        source.extractall(output)
    marker.touch()


def extract_idmt_guitar(archive: Path, root: Path) -> None:
    """Extract only the four guitar archives and skip bass/duplicate payloads."""

    output = root / "corpus" / "idmt-smt-audio-effects"
    marker = output / ".complete"
    if marker.exists():
        return
    staging = root / "downloads" / "idmt-smt-audio-effects" / "guitar-zips"
    staging.mkdir(parents=True, exist_ok=True)
    prefix = "IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/"
    names = (
        "Gitarre monophon.zip",
        "Gitarre monophon2.zip",
        "Gitarre polyphon.zip",
        "Gitarre polyphon2.zip",
    )
    with zipfile.ZipFile(archive) as outer:
        for filename in names:
            member = prefix + filename
            outer.extract(member, staging)
            nested = staging / member
            with zipfile.ZipFile(nested) as inner:
                inner.extractall(output)
    marker.touch()


def download_file(name: str, root: Path, entry: list) -> None:
    source = SOURCES[name]
    record = source["record"].rstrip("/").rsplit("/", 1)[-1]
    filename, size, expected = entry
    archive = root / "downloads" / name / filename
    url = f"https://zenodo.org/api/records/{record}/files/{filename}/content"
    print(f"download {name}/{filename}", flush=True)
    for attempt in range(2):
        fetch(url, archive, size)
        actual = digest(archive)
        if actual == expected:
            break
        archive.unlink(missing_ok=True)
        if attempt:
            raise RuntimeError(
                f"MD5 mismatch for {archive}: expected {expected}, got {actual}"
            )
    print(f"extract  {name}/{filename}", flush=True)
    if name == "idmt-smt-audio-effects":
        extract_idmt_guitar(archive, root)
    else:
        extract(archive, root / "corpus" / name / Path(filename).stem)


def download_git(name: str, root: Path) -> None:
    source = SOURCES[name]
    output = root / "corpus" / name
    revision = source["revision"]
    marker = output / ".complete"
    if marker.exists() and marker.read_text().strip() == revision:
        return
    environment = dict(os.environ)
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", source["repository"], str(output)],
            check=True,
            env=environment,
        )
    elif not (output / ".git").is_dir():
        raise RuntimeError(f"refusing to replace incomplete non-Git directory: {output}")
    subprocess.run(
        ["git", "-C", str(output), "fetch", "--depth", "1", "origin", revision],
        check=True,
        env=environment,
    )
    subprocess.run(
        ["git", "-C", str(output), "checkout", "--detach", revision],
        check=True,
        env=environment,
    )
    subprocess.run(["git", "-C", str(output), "lfs", "pull"], check=True)
    subprocess.run(["git", "-C", str(output), "lfs", "fsck"], check=True)
    actual = subprocess.run(
        ["git", "-C", str(output), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != revision:
        raise RuntimeError(f"revision mismatch for {name}: expected {revision}, got {actual}")
    marker.write_text(revision + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="*")
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--guitar-techs-expanded",
        action="store_true",
        help="download the P1/P2 chords, scales, and techniques used by v21+",
    )
    args = parser.parse_args()
    selected = args.sources or sorted(
        name for name, source in SOURCES.items() if not source.get("research_only")
    )
    unknown = sorted(set(selected) - set(SOURCES))
    if unknown:
        parser.error(f"unknown source: {', '.join(unknown)}")
    files = [
        (source, entry)
        for source in selected
        if SOURCES[source].get("method", "zenodo") == "zenodo"
        for entry in SOURCES[source]["files"]
        if args.full
        or entry[0] in PROTOTYPE
        or (
            args.guitar_techs_expanded
            and source == "guitar-techs"
            and entry[0] in EXTENDED_TECHS
        )
        or source in args.sources
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(download_file, source, args.root, entry)
            for source, entry in files
        ]
        futures.extend(
            executor.submit(download_git, source, args.root)
            for source in selected
            if SOURCES[source].get("method") == "git-lfs"
        )
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
