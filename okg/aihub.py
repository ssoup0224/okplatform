"""AIHub download via the official API shell (aihubshell), run on YOUR server.

AIHub data is access-restricted (approved account, Korean nationals), so it is downloaded
where you hold the approval — never inside this build environment. Official usage [R77]:
    curl -o "aihubshell" https://api.aihub.or.kr/api/aihubshell.do ; chmod +x aihubshell
    aihubshell -mode l -datasetkey 558                                  # list files + filekeys
    aihubshell -mode d -datasetkey 558 -filekey 1,2 -aihubapikey 'KEY'  # download selected files
The API key is read from the AIHUB_API_KEY environment variable and never printed or logged.
Only the label JSON files are selected by default (no need to download everything). [D42]
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
import urllib.request
import zipfile
from pathlib import Path

SHELL_URL = "https://api.aihub.or.kr/api/aihubshell.do"          # [R77]
LABEL_HINTS = ("라벨", "label", "TL_", "VL_", ".json")
SKIP_HINTS = ("원천", "source", "TS_", "VS_")


def ensure_shell(cache_dir: Path) -> Path:
    p = Path(cache_dir) / "aihub" / "aihubshell"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(SHELL_URL, tmp)
        os.replace(tmp, p)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return p


def list_files(shell: Path, datasetkey: int) -> str:
    """Raw listing text (file tree with sizes and filekeys) — shown to you in the notebook."""
    r = subprocess.run([str(shell), "-mode", "l", "-datasetkey", str(datasetkey)], capture_output=True, text=True,
                       timeout=300)
    return r.stdout + r.stderr


def parse_listing(listing: str) -> list[dict]:
    """Every line that ends with a numeric filekey -> {name, filekey, is_label}. The listing format
    is not formally documented, so the selection is printed for you to confirm or override. [D42]"""
    out = []
    for line in listing.splitlines():
        m = re.search(r"\|\s*(\d{3,})\s*$", line) or re.search(r"\s(\d{4,})\s*$", line)
        if not m:
            continue
        name = line[: m.start()].strip(" |├└─│")
        is_label = any(h.lower() in name.lower() for h in LABEL_HINTS) and not any(h.lower() in name.lower() for h in SKIP_HINTS)
        out.append({"name": name, "filekey": int(m.group(1)), "is_label": is_label})
    return out


PART_RE = re.compile(r"^(?P<base>.+)\.part(?P<off>\d+)$")


def merge_split_parts(directory: Path) -> list[Path]:
    """AIHub splits large archives into `<name>.zip.part0`, `<name>.zip.part<offset>`, ... that must be
    concatenated in numeric order before unzipping (the AIHub guide does this with `cat` on Linux/WSL).
    Parts are removed only after the merged file is complete. [R77][D62]"""
    groups: dict[Path, list[tuple[int, Path]]] = {}
    for p in Path(directory).rglob("*.part*"):
        m = PART_RE.match(p.name)
        if m and p.is_file():
            groups.setdefault(p.with_name(m.group("base")), []).append((int(m.group("off")), p))
    merged = []
    for target, parts in groups.items():
        parts.sort()
        tmp = target.with_name(target.name + ".merging")
        with open(tmp, "wb") as out:
            for _, part in parts:
                with open(part, "rb") as f:
                    while True:
                        chunk = f.read(1 << 24)
                        if not chunk:
                            break
                        out.write(chunk)
        os.replace(tmp, target)
        for _, part in parts:
            part.unlink()
        merged.append(target)
    return merged


def download(shell: Path, datasetkey: int, out_dir: Path, filekeys: list[int] | None = None) -> Path:
    """Download (all files, or only `filekeys`) into out_dir, merge split archives, then unzip .zip archives.
    Writes a _DONE marker only after everything finished, so an interrupted download is redone. [D1]"""
    from .config import _local_secret
    key = os.environ.get("AIHUB_API_KEY") or _local_secret("AIHUB_API_KEY")
    if not key:
        raise RuntimeError("no AIHub API key: set AIHUB_API_KEY or fill local_secrets.py")
    out_dir = Path(out_dir)
    done = out_dir / "_DONE"
    if done.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(shell.resolve()), "-mode", "d", "-datasetkey", str(datasetkey)]
    if filekeys:
        cmd += ["-filekey", ",".join(map(str, filekeys))]
    cmd += ["-aihubapikey", key]
    r = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stdout + r.stderr).replace(key, "***")
        raise RuntimeError(f"aihubshell failed (exit {r.returncode}):\n{msg[-2000:]}")
    merge_split_parts(out_dir)
    for z in list(out_dir.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():                     # names inside AIHub zips are often cp949
                try:
                    info.filename = info.filename.encode("cp437").decode("cp949")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
                zf.extract(info, z.parent / z.stem)
        z.unlink()
    done.write_text("ok")
    return out_dir
