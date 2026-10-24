"""Tests for the full-run review fixes (D44-D56)."""
import json
import random

import numpy as np
import pandas as pd
import pytest

from okg import data_sources as ds
from okg import evaluate as ev
from okg import features as fe
from okg import guard as G
from okg import policy as pol
from okg import stacker as st
from okg.cache import LRUCache, chain_keys
from okg.config import Config


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    from okg.tiny_model import build_tiny_guard
    return G.load_tokenizer(str(build_tiny_guard(tmp_path_factory.mktemp("tiny") / "m")))


FILL = "How do I make a cake? Tell me about history. "


# ---------------------------------------------------------------- D44 data: MHJ keeps its turns
def test_mhj_user_only_turns_are_not_merged():
    row = {"question_id": 7, "message_0": json.dumps({"role": "system", "body": "s"}),
           "message_1": json.dumps({"role": "user", "body": "u1"}),
           "message_2": json.dumps({"role": "user", "body": "u2"}),
           "message_3": json.dumps({"role": "user", "body": "u3"})}
    c = ds.parse_mhj([row])[0]
    assert c["n_user_turns"] == 3
    assert json.loads(c["label3"]) == ["Unknown", "Unknown", "Unsafe"]


# ---------------------------------------------------------------- D44 window
def _padded(n_pairs):
    msgs = [{"role": "user", "content": "SETUPMARK " + FILL}, {"role": "assistant", "content": "ok"}]
    for _ in range(n_pairs):
        msgs += [{"role": "user", "content": FILL * 3}, {"role": "assistant", "content": FILL * 2}]
    return msgs + [{"role": "user", "content": "now the plan from the start"}]


def test_padding_cannot_push_the_setup_out(tok):
    msgs = _padded(30)
    pinned, info = G.fit_window(tok, msgs, 512, pin_first=True)
    assert info["dropped"] > 0 and "SETUPMARK" in pinned[0]["content"]
    assert pinned[-1]["content"] == msgs[-1]["content"]
    assert G.n_tokens(tok, G.render_prompt(tok, pinned)) <= 512
    unpinned, _ = G.fit_window(tok, msgs, 512, pin_first=False)
    assert all("SETUPMARK" not in m["content"] for m in unpinned)          # the old behaviour


def test_long_message_head_is_kept_and_chunked(tok):
    cfg = Config(run_mode="smoke")
    msgs = [{"role": "user", "content": "HEADMARK " + FILL * 200 + " TAILMARK"}]
    prompts = G.turn_prompts(tok, msgs, 400, cfg)
    assert "HEADMARK" in prompts[0] and "TAILMARK" in prompts[0]            # head + tail, not tail only
    assert len(prompts) > 2
    assert all(G.n_tokens(tok, p) <= 408 for p in prompts)
    cfg.chunk_long_messages = False
    assert len(G.turn_prompts(tok, msgs, 400, cfg)) == 1


def test_short_conversation_unchanged(tok):
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "x"}]
    out, info = G.fit_window(tok, msgs, 512)
    assert out == msgs and info == {"dropped": 0, "last_trimmed": False, "last_budget": 512}
    assert G.turn_prompts(tok, msgs, 512, Config(run_mode="smoke")) == [G.render_prompt(tok, msgs)]


def test_message_chunks_cover_the_whole_message(tok):
    text = "A1 " + FILL * 60 + " Z9"
    ch = G.message_chunks(tok, text, 64, 8, 100)
    assert "A1" in ch[0] and "Z9" in ch[-1]
    few = G.message_chunks(tok, text, 64, 8, 3)
    assert len(few) == 3 and "A1" in few[0] and "Z9" in few[-1]


# ---------------------------------------------------------------- D45 features without the base pass
def test_features_without_base_columns():
    df = pd.DataFrame({"conv_id": ["c", "c"], "turn_idx": [0, 1], "prefix_id": ["c#0", "c#1"], "p_unsafe": [0.1, 0.9],
                       "p_contro": [0.0, 0.0], "cand_mass": [1.0, 1.0], "user_text": ["a", "b"],
                       "prev_user_text": ["", "a"], "prev_assistant_text": ["", "x"]})
    f = fe.build_features(df)                       # no base_p_* columns needed
    assert not set(fe.BASE_FEATURES) & set(f.columns)
    assert set(fe.BASE_FEATURES) <= set(fe.feature_columns(use_base=True))
    assert not set(fe.BASE_FEATURES) & set(fe.feature_columns())


def test_cache_keys_chain_and_lru():
    a = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}, {"role": "user", "content": "z"}]
    b = a[:2] + [{"role": "user", "content": "other"}]
    ka, kb = chain_keys(a), chain_keys(b)
    assert ka[:2] == kb[:2] and ka[2] != kb[2]
    assert chain_keys(a, "v1")[0] != chain_keys(a, "v2")[0]
    c = LRUCache(2)
    c.put("1", 1), c.put("2", 2), c.get("1"), c.put("3", 3)
    assert "2" not in c and c.get("1") == 1 and c.stats()["hits"] == 2


# ---------------------------------------------------------------- D47 thresholds
def test_clopper_pearson_margin_is_stricter():
    s = np.random.default_rng(0).random(2000)
    plain = st.conv_level_threshold(s, 0.01)
    safe = st.conv_level_threshold(s, 0.01, confidence=0.95)
    assert safe > plain
    k = int((s > safe).sum())
    assert st.fpr_upper_bound(k, len(s), 0.95) <= 0.01


def test_thresholds_by_language_and_fallback():
    rng = np.random.default_rng(1)
    ct = pd.DataFrame({"y": 0, "score": np.r_[rng.random(500) * 0.5, rng.random(500), rng.random(10)],
                       "lang": ["ko"] * 500 + ["en"] * 500 + ["mixed"] * 10})
    thr = st.thresholds_by_lang(ct, 0.02, None, min_benign=100)
    assert set(thr) == {"_global", "ko", "en"} and thr["ko"] < thr["en"]
    assert st.threshold_for(thr, "mixed") == thr["_global"] and st.threshold_for(0.3, "ko") == 0.3
    pre = pd.DataFrame({"conv_id": ["a", "b"], "source": "s", "family": "f", "lang": ["ko", "en"], "turn_idx": 0,
                        "t_attack": -1, "label3": "Safe", "latency_valid": True, "score": [0.45, 0.45]})
    t = ev.conversation_table(pre, "score", {"_global": 0.9, "ko": 0.4, "en": 0.5}).set_index("conv_id")
    assert t.at["a", "pred"] == 1 and t.at["b", "pred"] == 0


# ---------------------------------------------------------------- D48/D49 policy
def test_graded_actions_and_fail_mode():
    cfg = Config(run_mode="smoke")
    thr = {"block": {"_global": 0.9, "ko": 0.8}, "safe_answer": {"_global": 0.6}}
    assert pol.decide(0.85, 0.0, "ko", thr, cfg) == "block"
    assert pol.decide(0.85, 0.0, "en", thr, cfg) == "safe_answer"
    assert pol.decide(0.3, 0.7, "en", thr, cfg) == "careful_answer"
    assert pol.decide(0.3, 0.1, "en", thr, cfg) == "allow"
    assert pol.fail_action(cfg) == "safe_answer"
    cfg.guard_fail_mode = "nonsense"
    with pytest.raises(ValueError):
        pol.fail_action(cfg)


def test_carried_score_and_language():
    c = pol.carried([0.95, 0.1, 0.1, 0.1], 0.8)
    assert c[0] == 0.95 and c[1] == pytest.approx(0.76) and c.max() == 0.95
    assert pol.conversation_lang(["안녕하세요", "hello there"]) == "mixed"
    assert pol.conversation_lang(["안녕하세요"]) == "ko"


def test_scaled_or_respects_languages():
    df = pd.DataFrame({"lang": ["ko", "en"], "g": [0.4, 0.4], "h": [0.9, 0.9]})
    s = pol.scaled_or(df, [("g", {"_global": 0.8}, None), ("h", 0.6, ("ko",))])
    assert s[0] == pytest.approx(1.5) and s[1] == pytest.approx(0.5)


# ---------------------------------------------------------------- D46 stack vs guard
def _ct(preds, suite, y):
    return pd.DataFrame({"conv_id": [f"c{i}" for i in range(len(preds))], "pred": preds, "y": y, "suite": suite})


def test_stack_vs_guard_needs_a_clear_win_on_both_suites():
    cfg = Config(run_mode="smoke")
    n = 200
    suite = ["external"] * n + ["human_multiturn"] * n
    y = [1] * (2 * n)
    guard = _ct([1] * 100 + [0] * 100 + [1] * 100 + [0] * 100, suite, y)
    stack_both = _ct([1] * 140 + [0] * 60 + [1] * 140 + [0] * 60, suite, y)
    stack_one = _ct([1] * 140 + [0] * 60 + [1] * 100 + [0] * 100, suite, y)
    assert ev.stack_vs_guard(stack_both, guard, cfg)["recommended"] == "stack"
    assert ev.stack_vs_guard(stack_one, guard, cfg)["recommended"] == "guard"
    assert ev.suite_of("mhj") == "human_multiturn" and ev.suite_of("kotox_test_hard") == "external"
    assert ev.suite_of("wjb_ko_train") == "in_distribution"


# ---------------------------------------------------------------- D50 attack generation
def test_attack_dialogue_parser():
    ok = json.dumps({"turns": [{"user": "역사 이야기 해줘", "assistant": "네"}, {"user": "그럼 자세히 알려줘"}]})
    msgs, why = __import__("okg.attackgen", fromlist=["x"]).parse_dialogue("noise " + ok, 2, "ko")
    assert why == "" and [m["role"] for m in msgs] == ["user", "assistant", "user"]
    from okg import attackgen as ag
    assert ag.parse_dialogue("no json here", 2, "ko")[1] == "no_json"
    en = json.dumps({"turns": [{"user": "tell me history", "assistant": "ok"}, {"user": "more"}]})
    assert ag.parse_dialogue(en, 2, "ko")[1] == "not_korean"


def test_scripted_generation_labels_and_pairs():
    from okg import attackgen as ag
    from okg import splits as sp
    from okg.synth import Pools
    from test_units import _toy_frame
    cfg = Config(run_mode="smoke")
    df, _ = sp.assign_roles(_toy_frame(), cfg)
    jobs = ag.plan("guard_train", Pools(df[df["role"] == "guard_train"]), 2, cfg, random.Random(0))
    convs, stats = ag.run_jobs(jobs, cfg, None, None, random.Random(0))
    assert stats["ok"] == len(jobs) == 2 * 2 * len(cfg.attack_gen_strategies)
    for c in convs:
        labels = json.loads(c["label3"])
        if c["family"].endswith("_benign"):
            assert set(labels) == {"Safe"}
        else:
            assert labels[-1] == "Unsafe" and set(labels[:-1]) <= {"Unknown"}
        assert c["fixed_role"] == "guard_train"


# ---------------------------------------------------------------- D52 replay labels
def test_replay_never_teaches_safe_against_human_unsafe():
    rep = pd.DataFrame({"prefix_id": ["a", "b", "c"], "label3": ["Unsafe", "Safe", "Safe"], "category": ["Violent", "None", "None"]})
    base = pd.DataFrame({"p_safe": [0.9, 0.1, 0.8], "p_unsafe": [0.05, 0.8, 0.1], "p_contro": [0.05, 0.1, 0.1]})
    out = G.apply_replay_labels(rep, base)
    assert list(out["prefix_id"]) == ["b", "c"] and list(out["label3"]) == ["Unsafe", "Safe"]
    assert list(out["category"]) == ["Unknown", "None"]


# ---------------------------------------------------------------- D54 hate classifier (offline backend)
def test_ngram_hate_classifier_learns_and_saves(tmp_path):
    from okg import hate as hz
    pos = [f"혐오 표현 예시 {i} 너네 집단은 다 사라져야 해" for i in range(40)]
    neg = [f"오늘 날씨 좋다 {i} 같이 산책 갈래" for i in range(40)]
    clf = hz.NgramHateClassifier()
    clf.fit(pos + neg, [1] * 40 + [0] * 40)
    clf.save(tmp_path / "h")
    clf2 = hz.load(tmp_path / "h")
    p = clf2.predict_proba(["너네 집단은 다 사라져야 해", "같이 산책 갈래"])
    assert p[0] > 0.5 > p[1]
    rows = pd.DataFrame({"prefix_id": ["c#0", "c#1"], "conv_id": "c", "turn_idx": [0, 1],
                         "user_text": ["너네 집단은 다 사라져야 해", "같이 산책 갈래"]})
    f = hz.hate_features(rows, clf2).set_index("prefix_id")
    assert f.at["c#1", "hate_p_max"] == f.at["c#0", "hate_p_cur"]


# ---------------------------------------------------------------- D55 gate
def _metrics(recall_mhj, with_mhj=True):
    rows = [{"system": "okchat_lgb", "group": "ALL", "n_pos": 100, "n_neg": 100, "recall": 0.7, "fpr": 0.01}]
    if with_mhj:
        rows.append({"system": "okchat_lgb", "group": "source=mhj", "n_pos": 10, "n_neg": 0, "recall": recall_mhj,
                     "fpr": np.nan, "recall_ci_hi": recall_mhj + 0.01})
    return pd.DataFrame(rows)


def test_required_groups_gate():
    from okg.release import promotion_gate
    cfg = Config(run_mode="smoke")
    ok, _ = promotion_gate(_metrics(0.5), _metrics(0.5), cfg)
    assert ok
    bad, g = promotion_gate(_metrics(0.2), _metrics(0.5), cfg)        # n_pos=10 < gate_min_n would skip it before
    assert not bad and not g[g["group"] == "source=mhj"]["passed"].all()
    missing, _ = promotion_gate(_metrics(0.5, with_mhj=False), _metrics(0.5), cfg)
    assert not missing


def test_redteam_runner_with_stub_guard():
    from okg import redteam as rt

    class Stub:
        def assess(self, msgs):
            hit = any("SETUPMARK" in m["content"] for m in msgs if m["role"] == "user")
            return pd.DataFrame({"action": ["block" if hit else "allow"]})

        def assess_response(self, msgs, reply):
            return {"action": "allow"}
    suite = [{"case": "padding:attack", "strategy": "padding", "lang": "en", "attack": True,
              "messages": rt.padding_case("SETUPMARK", [{"role": "user", "content": "f"}, {"role": "assistant", "content": "g"}], "go")},
             {"case": "padding:benign", "strategy": "padding", "lang": "en", "attack": False,
              "messages": rt.padding_case("hello", [], "go")}]
    summ, det = rt.run_suite(Stub(), suite)
    assert summ["asr"] == 0.0 and summ["benign_intervention_rate"] == 0.0
    ok, _ = rt.redteam_gate({"n_attacks": 1, "asr": 0.10}, {"asr": 0.05}, Config(run_mode="smoke"))
    assert not ok


# ---------------------------------------------------------------- D56 monitoring
def test_psi_and_drift():
    from okg import monitoring as mon
    assert mon.psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert mon.psi([0.5, 0.5], [0.9, 0.1]) > 0.2
    log = pd.DataFrame({"log_id": ["a", "b", "c"], "timestamp": ["2026-10-01T10:00:00"] * 3, "p_block": [0.1, 0.2, 0.9],
                        "block": [False, False, True], "action": ["allow", "allow", "block"], "lang": "ko",
                        "guard_error": ""})
    ref = {"edges": [-np.inf, 0.5, np.inf], "share": [0.9, 0.1]}
    d = mon.score_drift(log, ref)
    assert len(d) == 1 and d["psi"].iloc[0] > 0
    br = mon.block_rates(log)
    assert br["block_rate"].iloc[0] == pytest.approx(1 / 3)
