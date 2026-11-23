"""WEBSPAM-UK2007 원본 파일 다운로드.

사용법:
    python scripts/download_data.py [--config configs/default.yaml]

- 원본 4개를 data/raw/에 저장하고, label tar.gz 2개는 data/raw/set1/, data/raw/set2/에 압축 해제한다.
- 이미 받은 파일은 다시 받지 않고 크기·SHA-256만 출력한다.
- 받는 중에는 `<파일명>.part`로 저장하고, 끝까지 받은 뒤에만 원래 이름으로 바꾼다.
- 출력된 크기·SHA-256은 data/README.md에 옮겨 적는다.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # `python scripts/...`로 실행해도 src를 import할 수 있게

from src.config import DEFAULT_CONFIG, load_config  # noqa: E402

# (기준 URL 기준 상대 경로, 압축 해제 폴더 이름 또는 None)
FILES = [
    ("webspam-uk2007-set1-1.0.tar.gz", "set1"),
    ("webspam-uk2007-set2-1.0.tar.gz", "set2"),
    ("links/uk-2007-05.hostgraph_weighted.graph-txt.gz", None),
    ("links/uk-2007-05.hostnames.txt.gz", None),
]

CHUNK = 1 << 20  # 1 MiB
USER_AGENT = "webspam-downloader/1.0"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """url을 dest.part로 받은 뒤 dest로 이름을 바꾼다. 실패하면 .part를 지운다."""
    part = dest.with_name(dest.name + ".part")
    # 서버가 "python"이 들어간 User-Agent를 406으로 거부하므로 일반 이름을 씀
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / total:6.1%}  {done:,} / {total:,} bytes", end="", flush=True)
                else:
                    print(f"\r  {done:,} bytes", end="", flush=True)
        print()
        if total and done != total:
            raise OSError(f"받은 크기 {done:,}가 Content-Length {total:,}와 다름")
        os.replace(part, dest)
    except BaseException:
        print()
        part.unlink(missing_ok=True)
        raise


def extract(archive: Path, out_dir: Path) -> None:
    """tar.gz를 out_dir에 푼다. 이미 내용이 있으면 건너뛴다."""
    if out_dir.is_dir() and any(out_dir.iterdir()):
        print(f"  압축 해제 건너뜀 (이미 있음): {out_dir.relative_to(ROOT)}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # filter="data": 절대 경로, 상위 폴더(..), 링크 등 out_dir 밖으로 풀리는 항목을 거부
        tar.extractall(out_dir, filter="data")
    print(f"  압축 해제 완료: {out_dir.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="WEBSPAM-UK2007 원본 다운로드")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="설정 파일 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_url = str(cfg.require("download", "base_url")).rstrip("/") + "/"
    raw_dir = cfg.path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for rel, extract_to in FILES:
        name = Path(rel).name
        dest = raw_dir / name
        url = base_url + rel
        print(f"[{name}]")
        if dest.exists():
            print("  다운로드 건너뜀 (이미 있음)")
        else:
            print(f"  다운로드: {url}")
            try:
                download(url, dest)
            except Exception as e:
                print(f"  실패: {e}", file=sys.stderr)
                return 1
        if extract_to:
            extract(dest, raw_dir / extract_to)
        summary.append((name, dest.stat().st_size, sha256_of(dest)))

    print("\n파일별 크기·SHA-256 (data/README.md에 기록)")
    print("| 파일 | 크기 (bytes) | SHA-256 |")
    print("|---|---:|---|")
    for name, size, digest in summary:
        print(f"| `{name}` | {size:,} | `{digest}` |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
