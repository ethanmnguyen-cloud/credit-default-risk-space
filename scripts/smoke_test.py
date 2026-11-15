"""After a deploy: wait until the live Space serves this commit, then test it in a browser.

Usage: python scripts/smoke_test.py <owner/space-name>

A green deploy step only proves the files were uploaded. This proves the live page
serves exactly the files that were tested, and that it scores correctly in a real
browser (scripts/browser_check.py).
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from browser_check import check  # noqa: E402

WAIT_SECONDS = 900
SHIPPED = ["index.html", "app.js", "content.js", "credit_model.js", "style.css", "model/model.json"]


def static_host(space: str) -> str:
    """Static Spaces are served from https://<owner>-<name>.static.hf.space."""
    owner, name = space.split("/")
    slug = re.sub(r"[^a-z0-9]+", "-", f"{owner}-{name}".lower()).strip("-")
    return f"https://{slug}.static.hf.space"


def fetch(url: str) -> bytes | None:
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError):
        return None


def wait_for_files(base: str) -> None:
    want = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SHIPPED}
    deadline = time.time() + WAIT_SECONDS
    while True:
        stamp = int(time.time())
        stale = [path for path, digest in want.items()
                 if (body := fetch(f"{base}/{path}?check={stamp}")) is None
                 or hashlib.sha256(body).hexdigest() != digest]
        if not stale:
            print(f"live files match this commit: {', '.join(SHIPPED)}")
            return
        if time.time() > deadline:
            raise SystemExit(f"After {WAIT_SECONDS} s the Space still serves old or missing files: {stale}")
        print(f"  waiting for {len(stale)} file(s): {', '.join(stale)}", flush=True)
        time.sleep(15)


def main(space: str) -> None:
    base = static_host(space)
    print(f"checking {base} (Space {space})")
    wait_for_files(base)
    check(base, screenshot="live_space.png")
    print("smoke test passed")


if __name__ == "__main__":
    main(sys.argv[1])
