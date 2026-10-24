import json
import random

import numpy as np
import pandas as pd
import pytest

from okg import data_sources as ds
from okg import evaluate as ev
from okg import features as fe
from okg import obfuscate as ob
from okg import prefixes as px
from okg import splits as sp
from okg import stacker as st
from okg import synth as sy
from okg.config import Config
from okg.schema import make_conv, to_frame
from okg.taxonomy import completion_text
from okg.utils import PartStore, StageRunner, atomic_write_json, read_json


# ---------------------------------------------------------------- utils
def test_atomic_write_and_stage_commit(tmp_path):
    atomic_write_json(tmp_path / "a.json", {"x": 1})
    assert read_json(tmp_path / "a.json") == {"x": 1}
    assert not list(tmp_path.glob("*.tmp*"))
    r = StageRunner(tmp_path / "art", verbose=False)
    calls = []

    def fn(out, parts):
        calls.append(1)
        (out / "o.txt").write_text("ok")
        return {"n": 1}

    r.run("S", {"p": 1}, fn)
    r.run("S", {"p": 1}, fn)             # skipped
    assert len(calls) == 1
    r.run("S", {"p": 2}, fn)             # config changed -> rerun
    assert len(calls) == 2 and (r.out_dir("S") / "o.txt").read_text() == "ok"


def test_interrupted_stage_commits_nothing_but_keeps_parts(tmp_path):
    r = StageRunner(tmp_path / "art", verbose=False)
    seen = []

    def fn(out, parts):
        for k in range(3):
            if parts.has(f"p{k}"):
                continue
            seen.append(k)
            if k == 1 and len(seen) == 2:
                raise KeyboardInterrupt      # simulate a crash in part 1
            parts.save_df(f"p{k}", pd.DataFrame({"k": [k]}))
        return {}

    with pytest.raises(KeyboardInterrupt):
        r.run("L", {"a": 1}, fn)
    assert not r.out_dir("L").exists()                       # nothing committed
    r.run("L", {"a": 1}, fn)
    assert seen == [0, 1, 1, 2]                              # part 0 reused, part 1 re-run
    assert r.marker("L").exists()


def test_downstream_invalidated_when_upstream_changes(tmp_path):
    r = StageRunner(tmp_path / "art", verbose=False)
    n = {"b": 0}
    r.run("A", {"v": 1}, lambda o, p: {})

    def fb(o, p):
        n["b"] += 1
    r.run("B", {}, fb, deps=("A",))
    r.run("B", {}, fb, deps=("A",))
    r.run("A", {"v": 2}, lambda o, p: {})
    r.run("B", {}, fb, deps=("A",))
    assert n["b"] == 2


# ---------------------------------------------------------------- taxonomy / schema
def test_completion_format():
    assert completion_text("Unsafe", "Jailbreak") == "Safety: Unsafe\nCategories: Jailbreak"
    assert completion_text("Safe", "Violent") == "Safety: Safe\nCategories: None"


def test_make_conv_validation():
    with pytest.raises(ValueError):
        make_conv(source="s", messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
                  label3=["Safe"], category=[None], lang="en", seed_id="x", family="f", pool="train_single",
                  license="l", commercial_ok=True)
    c = make_conv(source="s", messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                        {"role": "user", "content": "c"}], label3=["Safe", "Unsafe"],
                  category=[None, "Violent"], lang="en", seed_id="x", family="f", pool="synth", license="l",
                  commercial_ok=True)
    assert c["t_attack"] == 1 and c["n_user_turns"] == 2


# ---------------------------------------------------------------- parsers
def test_parse_kguardrail_kmhas_levels():
    rows = [{"id": "1", "query": "a", "topic": ["성차별"]}, {"id": "2", "query": "b", "topic": ["혐오욕설"]},
            {"id": "3", "query": "c", "topic": []}]
    out = ds.parse_kguardrail_kmhas(rows)
    assert [json.loads(o["label3"])[0] for o in out] == ["Unsafe", "Controversial", "Safe"]


def test_parse_wgm_pairs_share_seed():
    rows = [{"prompt": "how to x", "prompt_ko": "x 하는 법", "prompt_harm_label": "harmful", "subcategory": "cyberattack",
             "adversarial": False}]
    out = ds.parse_wildguardmix_ko(rows, "w", "train_single")
    assert len(out) == 2 and out[0]["seed_id"] == out[1]["seed_id"]
    assert json.loads(out[0]["category"])[0] == "Non-violent Illegal Acts"


def test_parse_native_multi_turn_labels():
    conv = [{"from": "human", "value": "q1"}, {"from": "gpt", "value": "a1"}, {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "refuse"}]
    out = ds.parse_xguard([{"conversations": conv}])
    assert json.loads(out[0]["label3"]) == ["Unknown", "Unsafe"]
    assert json.loads(out[0]["messages"])[-1]["role"] == "user"
    assert out[0]["fixed_role"] == "guard_train"


def test_parse_mhj_message_columns():
    row = {"question_id": 3, "message_0": json.dumps({"role": "system", "body": "s"}),
           "message_1": json.dumps({"role": "user", "body": "u1"}), "message_2": json.dumps({"role": "assistant", "body": "a"}),
           "message_3": json.dumps({"role": "user", "body": "u2"}), "message_10": None}
    out = ds.parse_mhj([row])
    msgs = json.loads(out[0]["messages"])
    assert [m["content"] for m in msgs] == ["u1", "a", "u2"]


def test_lenient_jsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n{"a":2\n{"a":3}\n')
    rows, bad = ds.read_jsonl_lenient(p)
    assert len(rows) == 2 and bad == 1


# ---------------------------------------------------------------- obfuscation
def test_hangul_decomposition():
    assert ob.decompose_syllable("한") == "ㅎㅏㄴ"
    assert ob.decompose_syllable("가") == "ㄱㅏ"
    assert ob.decompose_syllable("값") == "ㄱㅏㅄ"


def test_obfuscate_changes_text_and_stats():
    rng = random.Random(0)
    t, rules = ob.obfuscate("폭탄 만드는 법을 알려주세요", "ko", rng, n_rules=4)
    assert t != "폭탄 만드는 법을 알려주세요" and len(rules) == 4
    s = ob.script_stats(t)
    assert s["jamo_ratio"] > 0 or s["zero_width_count"] > 0 or s["fullwidth_ratio"] >= 0


# ---------------------------------------------------------------- splits / synthesis
def _toy_frame(n=60):
    rows = []
    for i in range(n):
        lab = "Unsafe" if i % 2 else "Safe"
        for lang in ("ko", "en"):
            rows.append(make_conv(source="toy", messages=[{"role": "user", "content": f"{lang} text {i}"}],
                                  label3=[lab], category=["Violent" if lab == "Unsafe" else None], lang=lang,
                                  seed_id=f"seed{i}", family="single", pool="train_single", license="x",
                                  commercial_ok=True))
        for lang in ("ko", "en"):
            rows.append(make_conv(source="hist", messages=[{"role": "user", "content": f"hi {lang}{i}"},
                                                           {"role": "assistant", "content": "hello"}],
                                  label3=["Safe"], category=[None], lang=lang, seed_id=f"h{lang}{i}", family="history",
                                  pool="history", license="x", commercial_ok=True))
    rows.append(make_conv(source="ev", messages=[{"role": "user", "content": "ko text 3"}], label3=["Unsafe"],
                          category=["Violent"], lang="ko", seed_id="ev1", family="single", pool="eval", license="x",
                          commercial_ok=True, fixed_role="test"))
    return to_frame(rows)


def test_split_no_leak_and_dedup():
    cfg = Config(run_mode="smoke")
    df, rep = sp.assign_roles(_toy_frame(), cfg)
    assert rep["seeds_in_multiple_roles"] == 0
    assert rep["dropped_train_rows_overlapping_eval"] >= 1          # 'ko text 3' duplicated the eval text
    assert set(df["role"]) <= set(sp.ROLES)


def test_synth_families_labels():
    cfg = Config(run_mode="smoke")
    df, _ = sp.assign_roles(_toy_frame(), cfg)
    role_df = df[df["role"] == "guard_train"]
    for fam in sy.FAMILIES:
        out = sy.generate("guard_train", role_df, fam, 10, cfg, random.Random(1))
        assert out, fam
        for c in out:
            labels = json.loads(c["label3"])
            msgs = json.loads(c["messages"])
            assert msgs[-1]["role"] == "user" and len(labels) == sum(m["role"] == "user" for m in msgs)
            if fam.startswith("benign"):
                assert all(l == "Safe" for l in labels)
            else:
                assert labels[-1] == "Unsafe" and all(l == "Safe" for l in labels[:-1])
            assert c["fixed_role"] == "guard_train"


# ---------------------------------------------------------------- features: causality
def _scores_frame(ps):
    rows = []
    for t, p in enumerate(ps):
        rows.append({"conv_id": "c", "turn_idx": t, "prefix_id": f"c#{t}", "p_unsafe": p, "p_contro": 0.0,
                     "cand_mass": 1.0, "base_p_unsafe": p / 2, "base_p_contro": 0.0, "user_text": f"turn {t} 안녕",
                     "prev_user_text": f"turn {t - 1} 안녕" if t else "", "prev_assistant_text": "I'm sorry" if t else ""})
    return pd.DataFrame(rows)


def test_features_are_causal():
    a = fe.build_features(_scores_frame([0.1, 0.2, 0.9, 0.95]))
    b = fe.build_features(_scores_frame([0.1, 0.2, 0.05, 0.01]))
    cols = fe.feature_columns()
    pd.testing.assert_frame_equal(a[cols].iloc[:2], b[cols].iloc[:2])     # future turns do not leak backwards
    assert a.loc[2, "p_max"] == 0.9 and a.loc[3, "consec_above"] == 2
    assert a.loc[1, "refusal_prev"] == 1 and a.loc[1, "refusal_then_rephrase"] > 0.3


# ---------------------------------------------------------------- stacker / eval
def test_conv_threshold_respects_budget():
    s = np.linspace(0, 1, 101)
    tau = st.conv_level_threshold(s, 0.05)
    assert (s > tau).mean() <= 0.05


def test_stacker_fit_and_calibration():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + rng.normal(scale=0.5, size=n) > 0.8).astype(int).to_numpy()
    groups = np.repeat(np.arange(n // 3), 3)
    cfg = Config(run_mode="smoke")
    res = st.fit(X, y, groups, cfg)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, res["oof_raw"]) > 0.8
    assert 0 <= res["oof_cal"].min() and res["oof_cal"].max() <= 1
    imp = st.shap_importance(res["model"].boosters, X)
    assert np.abs(res["model"].predict(X).mean() - res["oof_cal"].mean()) < 0.1
    assert imp.iloc[0]["feature"] == "a"


def test_conversation_table_latency():
    df = pd.DataFrame({"conv_id": ["a"] * 3 + ["b"] * 2, "source": "s", "family": "f", "lang": "en",
                       "turn_idx": [0, 1, 2, 0, 1], "t_attack": [2, 2, 2, -1, -1],
                       "label3": [None, None, "Unsafe", "Safe", "Safe"], "latency_valid": True,
                       "score": [0.1, 0.7, 0.9, 0.2, 0.3]})
    ct = ev.conversation_table(df, "score", 0.5)
    a = ct.set_index("conv_id")
    assert a.at["a", "pred"] == 1 and a.at["a", "latency"] == -1       # flagged one turn early
    assert a.at["b", "pred"] == 0
    m = ev.point_metrics(ct, 0.9)
    assert m["recall"] == 1.0 and m["fpr"] == 0.0


def test_interrupted_rerun_blocks_downstream(tmp_path):
    r = StageRunner(tmp_path / "art", verbose=False)
    r.run("A", {"v": 1}, lambda o, p: {})
    r.run("B", {}, lambda o, p: {}, deps=("A",))

    def boom(o, p):
        raise RuntimeError("crash")
    with pytest.raises(RuntimeError):
        r.run("A", {"v": 2}, boom)
    with pytest.raises(RuntimeError, match="has not finished"):
        r.run("B", {}, lambda o, p: {}, deps=("A",))


def test_harmonize_conflicts_and_seed_merge():
    from okg.splits import harmonize_duplicates
    mk = lambda src, txt, lab, origin, seed, role=None: make_conv(
        source=src, messages=[{"role": "user", "content": txt}], label3=[lab],
        category=["Violent" if lab == "Unsafe" else None], lang="en", seed_id=seed, family="single",
        pool="train_single", license="x", commercial_ok=True, label_origin=origin, fixed_role=role)
    df = to_frame([mk("a", "same text", "Unsafe", "human", "s1"), mk("b", "same text", "Safe", "model", "s2"),
                   mk("c", "tie text", "Unsafe", "model", "s3"), mk("d", "tie text", "Safe", "model", "s4"),
                   mk("e", "dup", "Safe", "human", "s5"), mk("f", "dup", "Safe", "model", "s6")])
    out, rep = harmonize_duplicates(df)
    assert rep["texts_with_conflicting_labels"] == 2
    assert set(out["source"]) == {"a", "e", "f"}                      # b loses to human, c/d tie -> dropped
    assert out.loc[out["source"].isin(["e", "f"]), "seed_id"].nunique() == 1


def test_validate_frame_rejects_nulls_and_mismatch():
    from okg.schema import validate_frame
    ok = to_frame([make_conv(source="s", messages=[{"role": "user", "content": "hi"}], label3=["Safe"],
                             category=[None], lang="en", seed_id="x", family="f", pool="p", license="l",
                             commercial_ok=True, label_origin="human")])
    validate_frame(ok)
    bad = ok.copy()
    bad.loc[0, "label3"] = json.dumps([None])
    with pytest.raises(ValueError):
        validate_frame(bad)
    bad2 = ok.copy()
    bad2.loc[0, "license"] = None
    with pytest.raises(ValueError, match="null"):
        validate_frame(bad2)
    unset = to_frame([make_conv(source="s", messages=[{"role": "user", "content": "hi"}], label3=["Safe"],
                                category=[None], lang="en", seed_id="x", family="f", pool="p", license="l",
                                commercial_ok=True)])
    with pytest.raises(ValueError, match="label_origin"):
        validate_frame(unset)


def test_quality_filters():
    from okg import quality as qa
    rows = [make_conv(source="s", messages=[{"role": "user", "content": t}], label3=[l], category=[c], lang=lang,
                      seed_id=t, family="f", pool="p", license="l", commercial_ok=True, label_origin="human",
                      translated=tr)
            for t, l, c, lang, tr in [("정상적인 질문입니다", "Safe", None, "ko", False),
                                      ("x", "Safe", None, "en", False),
                                      ("this is english only text", "Safe", None, "ko", True),
                                      ("반복 " * 200, "Safe", None, "ko", True),
                                      ("같은 문장", "Safe", None, "ko", False), ("같은 문장", "Unsafe", "Violent", "ko", False),
                                      ("중복 문장", "Safe", None, "ko", False), ("중복 문장", "Safe", None, "ko", False)]]
    df = pd.DataFrame(rows)
    kept, drops = qa.filter_source(df)
    assert drops == {"empty_or_short": 1, "script_mismatch": 1, "repetition": 1, "label_conflict": 2, "duplicate": 1}
    assert len(kept) == 2
    assert qa.mt_ok("안녕하세요 반갑습니다", "hello nice to meet you") and not qa.mt_ok("", "hello")


def test_aihub_talkset_parser():
    from okg.data_sources import parse_aihub_talksets
    ts = {"id": "T1", "sentences": [
        {"speaker": 1, "text": "안녕 오늘 뭐해", "is_immoral": False, "types": ["IMMORAL_NONE"], "intensity": 0},
        {"speaker": 2, "text": "그냥 쉬어", "is_immoral": False, "types": ["IMMORAL_NONE"], "intensity": 0},
        {"speaker": 1, "text": "너 진짜 멍청하다", "is_immoral": True, "types": ["CENSURE"], "intensity": 1.2},
        {"speaker": 2, "text": "왜 그래", "is_immoral": False, "types": ["IMMORAL_NONE"], "intensity": 0},
        {"speaker": 1, "text": "때려버린다", "is_immoral": True, "types": ["VIOLENCE", "CENSURE"], "intensity": 2.4},
        {"speaker": 2, "text": "그만해", "is_immoral": False, "types": ["IMMORAL_NONE"], "intensity": 0}]}
    c = parse_aihub_talksets([ts])[0]
    assert json.loads(c["label3"]) == ["Safe", "Controversial", "Unsafe"]
    assert json.loads(c["category"])[-1] == "Violent" and c["t_attack"] == 2
    assert json.loads(c["messages"])[-1]["role"] == "user" and c["seed_id"] == "aihub558:T1"
