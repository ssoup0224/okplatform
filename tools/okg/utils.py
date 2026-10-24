"""Checkpoint / resume utilities for okchat_guardrail.

Design goal (user requirement): a result is persisted ONLY when the unit of work that
produced it finished completely. An interrupted unit leaves nothing behind except a
temporary file/dir that is ignored and overwritten on the next run.

Reference tags such as [R9] point to REFERENCES.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import pandas as pd


# --------------------------------------------------------------------------------------
# Atomic file writes
# --------------------------------------------------------------------------------------
def _fsync_dir(path: Path) -> None:
    """fsync a directory so that a rename inside it is durable. [R9]"""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:  # e.g. platforms that cannot open directories
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write to <path>.tmp, flush+fsync, then os.replace onto <path>.

    os.replace is atomic on POSIX when src and dst are on the same filesystem [R9][R10];
    the temp file is therefore created in the destination directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())  # [R9] force bytes to disk before the rename
    os.replace(tmp, path)  # [R9] atomic rename
    _fsync_dir(path.parent)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8"))


def atomic_write_parquet(path: str | Path, df: pd.DataFrame) -> None:
    """Parquet via a temp file in the same directory, then os.replace. [R9][R43]"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_parquet(tmp, index=False)  # [R43]
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Config hashing (a changed config invalidates stale results)
# --------------------------------------------------------------------------------------
def stable_hash(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable object (sorted keys)."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Stage runner: whole-stage commit with a _DONE marker written LAST
# --------------------------------------------------------------------------------------
class StageRunner:
    """Runs named pipeline stages idempotently.

    Layout:  <artifacts>/<stage>/            final outputs (only exists if the stage finished)
             <artifacts>/<stage>/_DONE.json   marker (hash, timings) written last
             <artifacts>/<stage>.__tmp__/     scratch output dir while running
             <artifacts>/<stage>.__parts__/   completed sub-units ("turns") of a long stage

    A stage whose _DONE hash matches the current config hash is skipped. [D1]
    """

    def __init__(self, artifacts_dir: str | Path, verbose: bool = True):
        self.root = Path(artifacts_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    # ---- paths ------------------------------------------------------------------
    def out_dir(self, stage: str) -> Path:
        return self.root / stage

    def parts_dir(self, stage: str) -> Path:
        return self.root / f"{stage}.__parts__"

    def marker(self, stage: str) -> Path:
        return self.out_dir(stage) / "_DONE.json"

    # ---- status -----------------------------------------------------------------
    def done_hash(self, stage: str) -> str | None:
        m = self.marker(stage)
        if not m.exists():
            return None
        try:
            return read_json(m).get("hash")
        except Exception:
            return None

    def is_done(self, stage: str, h: str) -> bool:
        return self.done_hash(stage) == h

    def status(self) -> pd.DataFrame:
        rows = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / "_DONE.json").exists():
                info = read_json(d / "_DONE.json")
                rows.append({"stage": d.name, "hash": info.get("hash"),
                             "finished": info.get("finished"), "seconds": info.get("seconds")})
        return pd.DataFrame(rows)

    # ---- run --------------------------------------------------------------------
    def run(self, stage: str, params: dict, fn: Callable[[Path, "PartStore"], dict | None],
            deps: tuple[str, ...] = (), force: bool = False) -> Path:
        """Run `fn(tmp_out_dir, parts)` unless already done with the same hash.

        The hash covers `params` AND the hashes of upstream stages in `deps`, so that
        re-running an upstream stage invalidates everything downstream. [D1]
        """
        dep_hashes = {}
        for d in deps:
            dh = self.done_hash(d)
            if dh is None:
                raise RuntimeError(f"Stage '{stage}' depends on '{d}', which has not finished yet.")
            dep_hashes[d] = dh
        try:                                   # the stage body's own source is part of the hash [D1]
            import inspect
            fn_src = hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()[:12]
        except (OSError, TypeError):
            fn_src = "unavailable"
        h = stable_hash({"stage": stage, "params": params, "deps": dep_hashes, "fn": fn_src})
        final = self.out_dir(stage)
        if not force and self.is_done(stage, h):
            if self.verbose:
                print(f"[skip] {stage}: already finished with identical config (hash {h}).")
            return final

        # The old result no longer matches (or a re-run was forced): retract its marker first, so
        # downstream stages cannot run on it while this stage is incomplete. [D1]
        if self.marker(stage).exists():
            self.marker(stage).unlink()
        tmp = self.root / f"{stage}.__tmp__"
        if tmp.exists():
            shutil.rmtree(tmp)  # leftovers of an interrupted run are discarded
        tmp.mkdir(parents=True)
        parts = PartStore(self.parts_dir(stage), h, verbose=self.verbose)
        t0 = time.time()
        if self.verbose:
            print(f"[run ] {stage} (hash {h})")
        try:
            summary = fn(tmp, parts) or {}
        except BaseException:
            # Nothing is committed. Completed parts (if any) stay for the next attempt.
            if self.verbose:
                print(f"[FAIL] {stage}: nothing was committed. Re-run this cell to retry.")
            raise
        # Commit: replace the final dir with the finished tmp dir, then write marker.
        if final.exists():
            old = self.root / f"{stage}.__old__"
            if old.exists():
                shutil.rmtree(old)
            os.replace(final, old)  # [R9]
            os.replace(tmp, final)
            shutil.rmtree(old)
        else:
            os.replace(tmp, final)
        info = {"stage": stage, "hash": h, "params": params, "deps": dep_hashes,
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 2),
                "summary": summary}
        atomic_write_json(self.marker(stage), info)  # marker LAST [D1]
        parts.cleanup()
        if self.verbose:
            print(f"[done] {stage} in {info['seconds']}s  {json.dumps(summary, ensure_ascii=False, default=str)[:300]}")
        return final


class PartStore:
    """Per-part ("turn") persistence inside a long stage.

    Each part is written atomically together with its own marker, so an interrupted
    stage resumes at the first unfinished part. Parts written under a different config
    hash are ignored (and deleted). [D1][R9]
    """

    def __init__(self, directory: Path, stage_hash: str, verbose: bool = True):
        self.dir = Path(directory)
        self.hash = stage_hash
        self.verbose = verbose
        self.dir.mkdir(parents=True, exist_ok=True)
        info = self.dir / "_STAGE_HASH"
        if info.exists() and info.read_text().strip() != stage_hash:
            shutil.rmtree(self.dir)  # stale parts from an older config
            self.dir.mkdir(parents=True)
        atomic_write_bytes(info, stage_hash.encode())

    def path(self, name: str, ext: str = "parquet") -> Path:
        return self.dir / f"{name}.{ext}"

    def has(self, name: str) -> bool:
        return (self.dir / f"{name}.ok").exists()

    def save_df(self, name: str, df: pd.DataFrame) -> None:
        atomic_write_parquet(self.path(name), df)
        atomic_write_bytes(self.dir / f"{name}.ok", b"ok")  # marker after data

    def load_df(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(name))

    def save_json(self, name: str, obj: Any) -> None:
        atomic_write_json(self.path(name, "json"), obj)
        atomic_write_bytes(self.dir / f"{name}.ok", b"ok")

    def load_json(self, name: str) -> Any:
        return read_json(self.path(name, "json"))

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def run_optional(label: str, fn: Callable[[], Any]) -> Any:
    """Run an optional step; report failures without stopping the notebook."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"[optional step '{label}' failed — nothing committed] {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return None


@contextmanager
def timer(label: str):
    t0 = time.time()
    yield
    print(f"{label}: {time.time() - t0:.1f}s")


def code_hash(*module_names: str) -> str:
    """Hash of the source files a stage depends on, so editing the code invalidates the
    stage's cached result (config hashing alone would miss code changes). [D1]"""
    import importlib
    import inspect
    h = hashlib.sha256()
    for name in sorted(module_names):
        mod = importlib.import_module(f"okg.{name}")
        h.update(inspect.getsource(mod).encode("utf-8"))
    return h.hexdigest()[:12]
