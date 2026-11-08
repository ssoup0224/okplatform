"""Sources added 2026-10-28: XGuard cap, WildChat Korean, MTID (TurnGate), AIHub 71833 [D59]-[D62]."""
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from okg import aihub
from okg import data_sources as ds
from okg.config import Config

FIX = Path(__file__).parent / "fixtures"


def labels(c):
    return json.loads(c["label3"])


# ---------------------------------------------------------------- config
def test_xguard_cap_covers_whole_dataset():
    assert Config(run_mode="full").xguard_max_conversations >= 30695
    assert "aihub_71833" in Config(run_mode="full").hate_train_sources


def test_new_sources_registered():
    reg = {s.name: s for s in ds.REGISTRY}
    assert reg["wildchat_ko"].role == "history" and reg["wildchat_ko"].commercial_ok
    assert reg["mtid"].role == "train" and not reg["mtid"].commercial_ok
    assert reg["mtid_test"].role == "eval"
    assert reg["aihub_71833"].role == "train" and not reg["aihub_71833"].commercial_ok


# ---------------------------------------------------------------- WildChat
def test_parse_wildchat_filters_and_labels():
    conv = [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "네"},
            {"role": "user", "content": "날씨 어때"}, {"role": "assistant", "content": "맑아요"}]
    rows = [{"conversation_hash": "a", "language": "Korean", "toxic": False, "redacted": False, "conversation": conv},
            {"conversation_hash": "b", "language": "English", "toxic": False, "redacted": False, "conversation": conv},
            {"conversation_hash": "c", "language": "Korean", "toxic": True, "redacted": False, "conversation": conv},
            {"conversation_hash": "d", "language": "Korean", "toxic": False, "redacted": True, "conversation": conv},
            {"conversation_hash": "e", "language": "Korean", "toxic": False, "redacted": False, "conversation": conv[:2]}]
    out = ds.parse_wildchat(rows, min_user_turns=2)
    assert [c["seed_id"] for c in out] == ["wildchat:a"]
    c = out[0]
    assert c["pool"] == "history" and c["lang"] == "ko" and labels(c) == ["Safe", "Safe"]
    assert json.loads(c["messages"])[-1]["role"] == "assistant"      # history material may end on assistant


def test_wildchat_loader_uses_fixture(monkeypatch):
    monkeypatch.setenv("OKG_FIXTURE_DIR", str(FIX))
    cfg = Config(run_mode="smoke")
    out = ds.L_wildchat_ko(cfg)
    assert out and all(c["lang"] == "ko" and c["pool"] == "history" for c in out)
    assert all(sum(m["role"] == "user" for m in json.loads(c["messages"])) >= cfg.wildchat_min_user_turns for c in out)
    assert len(out) <= cfg.wildchat_ko_max_conversations


def test_wildchat_shard_scan_resumes_and_stops_early(tmp_path, monkeypatch):
    """The real (non-fixture) path: parquet shards, row-group language filter, per-shard cache."""
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq
    monkeypatch.delenv("OKG_FIXTURE_DIR", raising=False)
    src = tmp_path / "hub"
    src.mkdir()
    names = [f"data/train-{i:05d}-of-00003.parquet" for i in range(3)]
    for i, name in enumerate(names):
        rows = []
        for j in range(40):
            ko = j % 4 == 0
            rows.append({"conversation_hash": f"h{i}-{j}", "language": "Korean" if ko else "English", "turn": 2,
                         "toxic": False, "redacted": False, "model": "gpt-4o", "openai_moderation": [{"flagged": False}],
                         "conversation": [{"role": "user", "content": f"질문 {i}-{j}", "language": "Korean", "turn_identifier": 1},
                                          {"role": "assistant", "content": "답", "language": "Korean", "turn_identifier": 1},
                                          {"role": "user", "content": "또 질문", "language": "Korean", "turn_identifier": 2},
                                          {"role": "assistant", "content": "답2", "language": "Korean", "turn_identifier": 2}]})
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=10)
    calls = []

    def fake_download(repo, filename, repo_type, token, local_dir):
        calls.append(filename)
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((src / filename).read_bytes())
        return str(dest)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", lambda self, *a, **k: names + ["README.md"])
    cfg = Config(run_mode="smoke", project_dir=str(tmp_path))
    rows = ds.wildchat_korean_rows(cfg, need=15)          # 10 Korean per shard -> stops after 2 shards
    assert len(calls) == 2 and len(rows) == 20
    assert all(r["language"] == "Korean" and set(r["conversation"][0]) == {"role", "content"} for r in rows)
    cache = cfg.cache_dir / "wildchat_ko"
    assert len(list(cache.glob("*.ko.jsonl"))) == 2 and not list(cache.rglob("*.parquet"))   # shards deleted
    calls.clear()
    rows2 = ds.wildchat_korean_rows(cfg, need=15)         # second call: served from the per-shard cache
    assert calls == [] and len(rows2) == 20
    out = ds.parse_wildchat(rows2, min_user_turns=2)
    assert len(out) == 20 and all(c["lang"] == "ko" for c in out)


# ---------------------------------------------------------------- MTID
def _mtid_row(si, rid, k, extra_turns=0):
    conv = []
    for t in range(1, k + 1 + extra_turns):
        conv += [{"turn_id": t, "role": "user", "content": f"u{t}"}, {"turn_id": t, "role": "assistant", "content": f"a{t}"}]
    return {"sample_index": si, "rollout_id": rid, "target_turn": k, "meta_intent": "hack a server", "conversation": conv}


def test_parse_mtid_harmful_closure_turn():
    out = ds.parse_mtid([_mtid_row(7, 1, 3, extra_turns=2)], "harmful")
    c = out[0]
    msgs = json.loads(c["messages"])
    assert [m["content"] for m in msgs if m["role"] == "user"] == ["u1", "u2", "u3"]   # cut at the closure turn
    assert msgs[-1]["role"] == "user"
    assert labels(c) == ["Unknown", "Unknown", "Unsafe"] and c["t_attack"] == 2
    assert c["seed_id"] == "mtid:harmful:7" and c["family"] == "native_attack"
    assert json.loads(c["category"])[-1] == "Unknown"                    # filled by the teacher before SFT


def test_parse_mtid_benign_all_safe_and_rollout_cap():
    rows = [_mtid_row(3, rid, 2) for rid in (5, 1, 4, 2, 3)]
    out = ds.parse_mtid(rows, "benign", max_rollouts=2)
    assert len(out) == 2 and {c["seed_id"] for c in out} == {"mtid:benign:3"}
    assert all(labels(c) == ["Safe", "Safe"] and c["t_attack"] == -1 for c in out)
    assert all(c["family"] == "benign_native" and json.loads(c["messages"])[-1]["role"] == "user" for c in out)


def test_mtid_loaders_split_and_fixed_roles(monkeypatch):
    monkeypatch.setenv("OKG_FIXTURE_DIR", str(FIX))
    cfg = Config(run_mode="smoke")
    tr, te = ds.L_mtid_train(cfg), ds.L_mtid_test(cfg)
    assert tr and te
    assert not {c["seed_id"] for c in tr} & {c["seed_id"] for c in te}      # MTID's own seed split is kept
    assert all(c["fixed_role"] == "test" for c in te)
    per_seed = pd.Series([c["seed_id"] for c in tr]).value_counts()
    assert per_seed.max() <= cfg.mtid_rollouts_per_sample


# ---------------------------------------------------------------- AIHub 71833
@pytest.mark.parametrize("scores,expected", [
    ({"폭력위협/범죄조장": 1}, ("Unsafe", "Violent")),
    ({"종교": 3}, ("Unsafe", "Unethical Acts")),
    ({"종교": 1}, ("Controversial", "Unethical Acts")),
    ({"외설": 2}, ("Unsafe", "Sexual Content or Sexual Acts")),
    ({"외설": 1}, ("Controversial", "Sexual Content or Sexual Acts")),
    ({"모욕": 3}, ("Controversial", "Unethical Acts")),
    ({"욕설": 1}, ("Controversial", "Unethical Acts")),
    ({"정치성향": 2}, ("Controversial", "Politically Sensitive Topics")),
    ({"정상": 1}, ("Safe", None)),
    ({"정상": 0}, (None, None)),
    ({}, ("Safe", None)),
])
def test_aihub71833_label_mapping(scores, expected):
    assert ds.aihub71833_label(scores, unsafe_min=2) == expected


def test_aihub71833_header_matching_variants():
    korean = ["문장", "모욕", "욕설", "외설", "장애", "인종/지역", "연령", "종교", "정치성향", "직업", "성혐오", "폭력위협/범죄조장", "정상"]
    text, cats = ds.aihub71833_columns(korean)
    assert text == "문장" and len(cats) == 12 and cats["정치성향"] == "정치성향" and cats["성혐오"] == "성혐오"
    spaced = ["content", "모욕 ", "욕설", "외설", "장애", "인종 / 지역", "연령", "종교", "정치 성향", "직업", "성 혐오",
              "폭력 위협 / 범죄 조장", "정상"]
    text, cats = ds.aihub71833_columns(spaced)
    assert text == "content" and len(cats) == 12
    assert ds.aihub71833_columns(["content", "label"])[1] == {}          # stage-1 file is not stage-2


def test_aihub71833_unmapped_headers_raise():
    with pytest.raises(ValueError, match="could not map"):
        ds.parse_aihub71833(pd.DataFrame({"a": ["x"], "b": [1]}))


def test_aihub71833_loader_reads_stage2_only(monkeypatch):
    monkeypatch.setenv("OKG_FIXTURE_DIR", str(FIX))
    out = ds.L_aihub71833(Config(run_mode="smoke"))
    assert out and all(c["family"] == "comment" and c["lang"] == "ko" for c in out)
    got = {json.loads(c["label3"])[0] for c in out}
    assert got == {"Controversial", "Unsafe"}                         # 정상 half left out by default
    texts = [json.loads(c["messages"])[0]["content"] for c in out]
    assert not any("1단계" in t for t in texts)
    assert not any("''" in t or "#@" in t or "  " in t or "\n" in t for t in texts)
    cfg = Config(run_mode="smoke")
    cfg.aihub71833_keep_safe = True
    assert "Safe" in {json.loads(c["label3"])[0] for c in ds.L_aihub71833(cfg)}


def test_aihub71833_text_cleaning():
    assert ds.clean_aihub71833_text("저 ''놈'''들은  ''꺼져'''라 \n") == "저 놈들은 꺼져라"
    t = ds.clean_aihub71833_text("#@이름# 의원은 ''사퇴'''하라")
    assert t.split()[0] in ds.AIHUB71833_NAMES and t.endswith("의원은 사퇴하라")
    assert ds.clean_aihub71833_text("#@'''이름''# 왔다").split()[0] in ds.AIHUB71833_NAMES    # malformed tag
    assert ds.clean_aihub71833_text("링크 #@링크# 봐") is None                                 # other tags: dropped
    assert ds.clean_aihub71833_text("it's fine") == "it's fine"                                # single quote kept
    assert ds.clean_aihub71833_text("#@이름# x") == ds.clean_aihub71833_text("#@이름# x")      # deterministic


def test_aihub71833_cp949_csv(tmp_path):
    df = pd.DataFrame({"문장": ["욕하는 문장"], "모욕": [2], "욕설": [0], "외설": [0], "장애": [0], "인종/지역": [0],
                       "연령": [0], "종교": [0], "정치성향": [0], "직업": [0], "성혐오": [0], "폭력위협/범죄조장": [0], "정상": [0]})
    p = tmp_path / "s2.csv"
    df.to_csv(p, index=False, encoding="cp949")
    out = ds.parse_aihub71833(ds.read_csv_any(p))
    assert labels(out[0]) == ["Controversial"]


def test_aihub_merge_split_parts(tmp_path):
    import zipfile
    z = tmp_path / "src.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("data/stage2.csv", "문장,모욕\nx,1\n" * 2000)
    blob = z.read_bytes()
    z.unlink()
    d = tmp_path / "dl" / "sub"
    d.mkdir(parents=True)
    cut = [0, 700, 1500, len(blob)]
    for a, b in zip(cut, cut[1:]):                       # offsets as part suffixes, written out of order
        (d / f"TL_1.zip.part{a}").write_bytes(blob[a:b])
    merged = aihub.merge_split_parts(tmp_path / "dl")
    assert merged == [d / "TL_1.zip"] and (d / "TL_1.zip").read_bytes() == blob
    assert not list(d.glob("*.part*"))
    with zipfile.ZipFile(d / "TL_1.zip") as zf:
        assert zf.namelist() == ["data/stage2.csv"]


# ---------------------------------------------------------------- pipeline wiring
def test_translation_takes_mtid_attacks_not_benign_twins():
    src = Path(ds.__file__).with_name("pipeline.py").read_text(encoding="utf-8")
    assert '"mtid"]' in src and 'convs["t_attack"] >= 0' in src
