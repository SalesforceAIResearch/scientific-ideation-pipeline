"""Data preparation for the ideation-v0 stages (RL/GRPO and SFT).

Turns a scored ideation JSONL (e.g.
``final_data_jun22_v0_insight/train_final_opus_48_scored_jun23.jsonl`` for RL, or
``final_data_jun22_v0_insight/final_sft_with_inputs.jsonl`` for SFT) into verl
parquet splits.

The prompt is *built*, not read from a column: each record's per-source summaries
are assembled into the exact live-inference prompt via ``prompts.build_prompt``
(reusing ``inference.dataloader.build_input_fields``), so training input ==
inference input -- for BOTH stages. (The SFT JSONL also ships a stored ``query``,
but it was built with an older prompt template -- 93 chars / one criterion bullet
short of the live prompt -- so we rebuild rather than trust it. This keeps the SFT
prompt byte-identical to the RL/inference prompt.)

    # RL data (verl PPO schema) for GRPO:
    python prepare_ideation_data.py --stage rl \
        --in  .../train_final_opus_48_scored_jun23.jsonl \
        --test-in .../test_data_2026_jun23.jsonl \
        --out-dir $STORAGE_DIR/grpo_ideation_v0 \
        --n-val 100 --num-ideas 5

    # SFT data (messages schema) for the format-acquisition warm-start:
    python prepare_ideation_data.py --stage sft \
        --in  .../final_sft_with_inputs.jsonl \
        --out-dir $STORAGE_DIR/sft_ideation_v0 \
        --n-val 100 --num-ideas 5 --max-seq-len 11776 --require-valid-json

Splits
------
``--in`` is train-only; we slice a deterministic, leakage-free holdout from it:
``val`` = the LAST ``n_val`` usable rows, ``train`` = everything before it (or
``n_train`` if given). No shuffle, so train/val never overlap and are stable
across reruns. ``--test-in`` (same schema, a separate held-out file) is written
WHOLE to ``test.parquet`` -- no slicing, no holdout. Pass either or both.
(``--test-in`` is RL-only; SFT has no test split.)

RL schema (per row), consumed by verl ``main_ppo`` + ``rewards/f9b_v0.py``:
  data_source                       task tag
  prompt        [{role:user, content: <built ideation prompt>}]
  ability       "insight"
  reward_model  {style:"rule", ground_truth: <target_idea_str>}
  extra_info    {index, split, query, target_arxiv_id, target_title,
                 num_documents, num_ideas,
                 source_summaries: <JSON string of [{title, summary:{...}}]>}

SFT schema (per row), consumed by verl ``sft_trainer`` (MultiTurnSFTDataset):
  messages      [{role:user, content: <built ideation prompt>},
                 {role:assistant, content: "<think>..</think>\n<answer>[..]</answer>"}]
  target_arxiv_id, num_documents, num_ideas    carried for traceability

``source_summaries`` (RL only) is written as a JSON STRING (not a nested list) so
pyarrow never has to infer a ragged nested schema across rows; the reward
json-loads it. It carries the per-source {problem, insight, motivation} cards so
the F9-B novelty judge can ground each generated idea against the real source
documents (3-16 of them), not just two abstracts.

Schema versions (``--prompt-type``)
-----------------------------------
``--prompt-type`` selects which ideation template and which source/gold columns to
read, so ONE script serves both dataset versions:

  ideation_v0 (default)  v0 cards {motivation, problem, insight{claim, grounding}}
                         read from ``input_full_summaries``; gold from
                         ``target_idea_str``.
  ideation_si            v1/SI cards {problem_gap, core_claim, method, inspiration}
                         read from ``input_si_summaries_str``; gold rendered from
                         ``target_si_idea_grounded``.

It defaults to the v0 template, so every existing invocation produces
byte-identical output. The per-format column mapping is NOT re-implemented here:
``inference.dataloader.build_input_fields(record, fmt)`` already owns it (fmt "v0"
vs "si"), and ``prompts.PROMPT_ANSWER_FORMAT`` already maps prompt_type -> fmt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root on the path so `prompts` / `inference` import whether this is run
# from the repo root or the data_prep dir; grpo_training too so `common` resolves
# (the SI gold renderer imports common.ideas_si).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_GRPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference.dataloader import _iter_jsonl, build_input_fields, read_jsonl  # noqa: E402
from prompts import DEFAULT_NUM_IDEAS, DEFAULT_PROMPT_TYPE, build_prompt  # noqa: E402
from prompts.ideation import PROMPT_ANSWER_FORMAT, PROMPT_TYPES  # noqa: E402

DATA_SOURCE = "ideation_v0"
ABILITY = "insight"

# Fields of each per-source summary card we hand to the novelty judge (it reads
# summary.problem / .insight / .motivation; see common/novelty.card_block).
_SUMMARY_FIELDS = ("problem", "insight", "motivation")

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

# --------------------------------------------------------------------------- #
# Streaming reader (see --stream)
# --------------------------------------------------------------------------- #
# Above this input size, --stream=auto switches from read_jsonl to _iter_jsonl.
# 256 MB is well above every v0 file and well below the v1/SI ones, so v0 keeps
# the historical code path byte for byte and v1 never hits the blow-up below.
_STREAM_AUTO_BYTES = 256 * 1024 * 1024

# Per-record fields DROPPED while streaming. These are full paper BODIES: 23 KB
# +23 KB of a ~70 KB v1 row, i.e. ~65% of the bytes, and NOTHING in this file or
# its two dependencies reads them. Verified by enumerating every record.get() in
# prepare_ideation_data.py + inference/dataloader.py + common/ideas_si.py: the
# prep touches only input_titles, input_full_summaries[_str],
# input_si_summaries_str, the four target_*/gold fields, target_arxiv_id,
# target_title, and the SFT trio (completion / think_str / answer_str).
#
# IF A FUTURE REWARD NEEDS DOCUMENT BODIES, REMOVE THEM FROM THIS TUPLE -- do not
# work around it downstream. The drop is why streaming's retained footprint is
# small; without it we stream the peak down but still hold every body in RAM.
_STREAM_DROP_FIELDS = ("input_full_documents", "target_full_document")


def _should_stream(path: Path, mode: str) -> bool:
    """Whether to read `path` one record at a time. mode is "auto" | "1" | "0".

    Keyed on FILE SIZE, not on the cluster: the v1/SI files are big because each
    row carries full document bodies (the SI schema), and they are that size on
    any machine. Gating on a machine profile instead would silently restore the
    blow-up whenever v1 data is prepped somewhere else, and would make small v0
    files pay for streaming here."""
    if mode == "1":
        return True
    if mode == "0":
        return False
    try:
        return path.stat().st_size > _STREAM_AUTO_BYTES
    except OSError:
        return False


def _read_rows(path: Path, stream_mode: str, tag: str) -> List[Dict[str, Any]]:
    """Load `path` as a list of records, streaming + projecting when it is large.

    WHY THIS EXISTS. read_jsonl (inference/dataloader.py:50) holds the ENTIRE file
    as one str AND the list of split substrings AND the parsed dicts, all live at
    once. Its own docstring measures ~8.0 GB RSS on the 730 MB v0 test split; the
    v1 train_gold.jsonl is 2.25 GB, which extrapolates to ~20-25 GB for one call
    -- enough to stall a shared login node on page-cache churn alone, with no
    output written because the first print happens after the read returns.

    Streaming yields one record at a time and drops _STREAM_DROP_FIELDS before the
    dict is retained, so peak and retained footprint both stay small.

    IDENTICAL PARSE, NOT MERELY SIMILAR: _iter_jsonl's line splitting is verified
    record-for-record equal to read_jsonl's (see its docstring on universal
    newlines). The ONLY difference in the returned rows is the absent body fields,
    which nothing here reads."""
    if not _should_stream(path, stream_mode):
        return read_jsonl(path)
    drop = set(_STREAM_DROP_FIELDS)
    rows = [{k: v for k, v in rec.items() if k not in drop}
            for rec in _iter_jsonl(path)]
    mb = path.stat().st_size / (1024 * 1024)
    print(f"[{tag}] streamed {path.name} ({mb:,.0f} MB), dropped per-record fields: "
          f"{', '.join(_STREAM_DROP_FIELDS)}")
    return rows


def _answer_fmt(prompt_type: str) -> str:
    """The answer/summary format ("v0" | "si") a prompt_type implies. Single source
    of truth is prompts.PROMPT_ANSWER_FORMAT."""
    return PROMPT_ANSWER_FORMAT.get(prompt_type, "v0")


def _data_source(prompt_type: str) -> str:
    """Task tag written to every row. v0 keeps the historical "ideation_v0" string
    verbatim so existing parquet / reward routing is unaffected."""
    return DATA_SOURCE if _answer_fmt(prompt_type) == "v0" else "ideation_si"


def _build_query(record: Dict[str, Any], num_ideas: int,
                 prompt_type: str = DEFAULT_PROMPT_TYPE) -> Tuple[str, int]:
    """Rebuild the live prompt from a record's source summaries. Returns
    (prompt, num_documents). Shared by both stages so SFT prompt == RL prompt.

    `prompt_type` picks the template AND (via PROMPT_ANSWER_FORMAT) which
    per-source summary column build_input_fields reads, so the documents the model
    is shown always match the schema it is asked to emit."""
    input_fields = build_input_fields(record, _answer_fmt(prompt_type))
    prompt = build_prompt(input_fields, prompt_type=prompt_type, num_ideas=num_ideas)
    return prompt, len(input_fields)


def _source_summaries(record: Dict[str, Any],
                      prompt_type: str = DEFAULT_PROMPT_TYPE) -> List[Dict[str, Any]]:
    """Per-source cards aligned to the SAME documents the prompt was built from:
    title + {problem, insight, motivation}, dropping sources whose summary is
    None (matches inference.dataloader.build_input_fields).

    EMPTY FOR ANY NON-v0 FORMAT. This column exists only for the older
    novelty-bearing rewards (rewards/f9b_v0.py, f9b_v1.py), which ground each
    generated idea against the real source cards via common.novelty. The v6 family
    dropped novelty entirely -- f9b_impdist_len_v6.py:94 does not even import
    common.novelty, and its `novelty_mean` output is a logged-only 0.0 constant --
    so an SI translation would be dead weight written into every row.

    TODO: IF NOVELTY COMES BACK, FILL THIS IN FOR SI FIRST. A novelty-bearing
    reward run against SI data would NOT crash on the empty list -- f9b_v0.py
    _get_documents falls through to the abstract_N and 'Paper N:' regex paths and
    then scores novelty against zero documents, so you get a plausible-looking
    number computed from nothing. To fill it: read ``input_si_summaries`` and map
    the slots the way common/ideas_si.py does (problem_gap -> problem, core_claim
    -> insight, method -> motivation) into the same {id, title, summary} shape, and
    drop the early return below. Raising instead of returning [] would be the safer
    guard, but data_prep/darling_k1/prep_darling_k1_data.py:144 calls this with no
    prompt_type and relies on the v0 default, so it must stay callable."""
    if _answer_fmt(prompt_type) != "v0":
        return []
    titles = record.get("input_titles") or []
    summaries = record.get("input_full_summaries") or []
    out: List[Dict[str, Any]] = []
    for i, (title, summ) in enumerate(zip(titles, summaries)):
        if summ is None:
            continue
        card = {f: str((summ or {}).get(f, "") or "").strip() for f in _SUMMARY_FIELDS}
        out.append({
            "id": f"doc{i}",
            "title": str(title or "").strip() or f"Document {i + 1}",
            "summary": card,
        })
    return out


def _ground_truth(record: Dict[str, Any], prompt_type: str) -> str:
    """The gold idea string the impact judge compares a candidate against.

    v0: the stored ``target_idea_str``, unchanged.

    SI: RE-RENDERED to the 3 judged fields by common.ideas_si.render_si_card --
    the same slot renderer the reward runs on candidates. The stored gold carries
    all FOUR fields including ``inspiration``, and the candidate card deliberately
    excludes it, so handing the judge the stored string verbatim would mean a
    4-field gold against a 3-field candidate: a systematic format asymmetry on the
    one comparison the whole reward turns on.

    Which column holds SI gold depends on the file, so try all three in order:
    ``target_si_idea_grounded`` (dict; train_gold/test), then the two flat labeled
    strings. ``train_multi`` -- the RL train source -- ships ONLY
    ``target_idea_str``, in SI field order despite the v0-looking name, which is
    why the renderer accepts a string as well as a dict."""
    if _answer_fmt(prompt_type) == "v0":
        return str(record.get("target_idea_str") or "").strip()
    from common.ideas_si import render_si_card  # local: keeps the v0 path import-light

    for key in ("target_si_idea_grounded", "target_si_idea_grounded_str",
                "target_idea_str"):
        rendered = render_si_card(record.get(key)).strip()
        if rendered:
            return rendered
    return ""


def to_rl_record(record: Dict[str, Any], split: str, idx: int, num_ideas: int,
                 prompt_type: str = DEFAULT_PROMPT_TYPE) -> Dict[str, Any]:
    prompt, num_documents = _build_query(record, num_ideas, prompt_type)
    extra = {
        "index": idx,
        "split": split,
        "query": prompt,
        "target_arxiv_id": record.get("target_arxiv_id"),
        "target_title": record.get("target_title"),
        "num_documents": num_documents,
        "num_ideas": num_ideas,
        # JSON string -> stable flat parquet schema; reward json-loads it.
        "source_summaries": json.dumps(_source_summaries(record, prompt_type),
                                       ensure_ascii=False),
    }
    return {
        "data_source": _data_source(prompt_type),
        "prompt": [{"role": "user", "content": prompt}],
        "ability": ABILITY,
        "reward_model": {"style": "rule",
                         "ground_truth": _ground_truth(record, prompt_type)},
        "extra_info": extra,
    }


def _completion(record: Dict[str, Any]) -> str:
    """Reassemble the assistant turn as the policy must emit it at RL/inference:
    <think>{think}</think>\n<answer>{answer}</answer>.

    Prefer the stored ``completion`` (already in that exact shape) when present;
    otherwise rebuild from ``think_str`` + ``answer_str``. We do NOT rebuild the
    answer JSON -- it is taken verbatim so the supervised target matches the
    reward's strict ``json.loads`` expectations byte-for-byte."""
    comp = record.get("completion")
    if isinstance(comp, str) and "<answer>" in comp and "</answer>" in comp:
        return comp.strip()
    think = str(record.get("think_str") or "").strip()
    answer = str(record.get("answer_str") or "").strip()
    return f"<think>{think}</think>\n<answer>{answer}</answer>"


def to_sft_record(record: Dict[str, Any], split: str, idx: int, num_ideas: int,
                  messages_key: str,
                  prompt_type: str = DEFAULT_PROMPT_TYPE) -> Dict[str, Any]:
    prompt, num_documents = _build_query(record, num_ideas, prompt_type)
    completion = _completion(record)
    return {
        messages_key: [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "target_arxiv_id": record.get("target_arxiv_id"),
        "num_documents": num_documents,
        "num_ideas": num_ideas,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["rl", "sft"], default="rl",
                   help="Output schema: 'rl' (verl PPO) or 'sft' (verl messages). Default rl.")
    p.add_argument("--in", dest="in_path", type=Path, default=None,
                   help="Source scored ideation JSONL for train/val.")
    p.add_argument("--test-in", dest="test_in_path", type=Path, default=None,
                   help="Held-out test JSONL (RL only). Written whole to test.parquet (no slicing).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output dir for train.parquet / val.parquet / test.parquet.")
    p.add_argument("--n-train", type=int, default=0,
                   help="Train rows (default 0 = all usable rows before the val tail).")
    p.add_argument("--n-val", type=int, default=0,
                   help="Val rows, taken as the LAST n_val usable rows (leakage-free holdout).")
    p.add_argument("--min-documents", type=int, default=2,
                   help="Keep records with strictly MORE than this many usable source "
                        "summaries (default 2 -> >=3; matches inference.dataloader).")
    p.add_argument("--num-ideas", type=int, default=DEFAULT_NUM_IDEAS,
                   help=f"Ideas the prompt asks for / K (default {DEFAULT_NUM_IDEAS}).")
    p.add_argument("--prompt-type", choices=sorted(PROMPT_TYPES),
                   default=DEFAULT_PROMPT_TYPE,
                   help=f"Ideation template + card schema (default {DEFAULT_PROMPT_TYPE}). "
                        "'ideation_si' switches to the v1/SI 4-field schema: SI source "
                        "cards in, SI gold re-rendered to its 3 judged fields, "
                        "data_source='ideation_si'.")
    # --- SFT-only knobs ---
    p.add_argument("--messages-key", default="messages",
                   help="SFT messages column name (default 'messages').")
    p.add_argument("--max-seq-len", type=int, default=0,
                   help="SFT only: drop rows whose (prompt+completion) exceeds this many "
                        "tokens (0 = no filter). Prevents right-truncation of <answer>. "
                        "Uses --tokenizer.")
    p.add_argument("--tokenizer", default="allenai/Olmo-3-7B-Think-SFT",
                   help="HF tokenizer for --max-seq-len measurement.")
    p.add_argument("--repair-json", action="store_true",
                   help="SFT: before the valid-JSON filter, rewrite <answer> payloads "
                        "that fail json.loads ONLY because of a trailing comma. "
                        "Recovers 729 of the 734 rejected v1/SI rows (9.1% of that "
                        "corpus). OPT-IN so existing prepped dirs still rebuild "
                        "byte-identically -- see repair_answer_json().")
    p.add_argument("--require-valid-json", action="store_true",
                   help="SFT only: drop (and count) rows whose <answer> is not valid JSON "
                        "with exactly --num-ideas ideas. The supervised target must be clean.")
    p.add_argument("--stream", choices=["auto", "1", "0"], default="auto",
                   help="Read the input one record at a time and drop unused full-document "
                        f"body fields. 'auto' (default) streams inputs over "
                        f"{_STREAM_AUTO_BYTES // (1024 * 1024)} MB -- which is every v1/SI "
                        "file and no v0 file, so v0 output is unchanged. '1'/'0' force it. "
                        "Non-streaming holds the whole file + its split + the parsed rows "
                        "live at once (~10x the file size in RSS).")
    return p.parse_args()


def _warn_empty_ground_truth(records: List[Dict[str, Any]], tag: str) -> None:
    """Count rows whose ground_truth came out empty. The impact judge scores each
    candidate card AGAINST this string, so an empty one silently makes that row's
    impact signal meaningless rather than failing loudly. Two v1 test rows ship an
    empty gold upstream, so this is a real (if tiny) case -- report it instead of
    dropping the row, which would desync row counts from the inference test set."""
    n = sum(1 for r in records
            if not str((r.get("reward_model") or {}).get("ground_truth") or "").strip())
    if n:
        print(f"[{tag}] WARNING: {n} row(s) have an EMPTY ground_truth "
              f"(impact judge has nothing to compare against for these).")


def _write(records: List[Dict[str, Any]], out_path: Path) -> None:
    import datasets

    if records and "reward_model" in records[0]:
        _warn_empty_ground_truth(records, out_path.stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(records).to_parquet(str(out_path))
    print(f"wrote {out_path} ({len(records)} rows)")


def _usable(rows: List[Dict[str, Any]], min_documents: int, tag: str,
            prompt_type: str = DEFAULT_PROMPT_TYPE) -> List[Dict[str, Any]]:
    """Keep rows with strictly more than `min_documents` usable source summaries
    (same filter the live dataloader applies), preserving order.

    Counted in the SELECTED format: v0 and SI read different per-source columns, so
    a row usable in one may be empty in the other."""
    fmt = _answer_fmt(prompt_type)
    keep = [r for r in rows if len(build_input_fields(r, fmt)) > min_documents]
    print(f"[{tag}] usable rows (> {min_documents} docs): {len(keep)} (dropped {len(rows) - len(keep)})")
    return keep


def _extract_answer(completion: str) -> Optional[str]:
    m = _ANSWER_RE.search(completion)
    return m.group(1).strip() if m else None


# A comma immediately before a closing brace/bracket. Legal in JS, rejected by
# json.loads, and present in 729 of the 8,006 v1/SI corpus rows -- 9.1% of the
# corpus was being discarded by --require-valid-json over this alone.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def repair_answer_json(record: Dict[str, Any], num_ideas: int) -> bool:
    """Rewrite a record's <answer> into strict JSON in place. True if it changed.

    WHY REPAIR RATHER THAN RELAX THE FILTER. Loosening the parse and training on
    the payload as-is would teach the policy to emit trailing commas -- and the
    reward's format gate does a strict json.loads, so every such rollout takes a
    terminal 0. Repairing gets the row AND a target that is valid under the reward.

    CONSERVATIVE BY CONSTRUCTION: a row is rewritten only if it does NOT parse now
    and the repaired text parses to a list of exactly `num_ideas` ideas. So repair
    can never alter a row that was already fine, and can never smuggle a malformed
    row past the filter -- anything it cannot fix is left untouched and still
    dropped. Verified on the 115 selected strict-win cards: 114 repaired to exactly
    3 ideas with the SI key set, 1 (2305.18917, an empty `""` key with no value)
    correctly left broken.

    Both `answer_str` and `completion` are rewritten because _completion() PREFERS
    the stored `completion` -- fixing only answer_str would leave the supervised
    target untouched on every row that has one."""
    raw_comp = _completion(record)
    ans = _extract_answer(raw_comp)
    if ans is None:
        return False
    try:
        json.loads(ans)
        return False                      # already valid -- never touch it
    except Exception:
        pass
    fixed = _TRAILING_COMMA_RE.sub(r"\1", ans)
    if fixed == ans:
        return False
    try:
        ideas = json.loads(fixed)
    except Exception:
        return False
    if not isinstance(ideas, list) or len(ideas) != num_ideas:
        return False
    record["answer_str"] = fixed
    comp = record.get("completion")
    if isinstance(comp, str) and "<answer>" in comp and "</answer>" in comp:
        record["completion"] = _ANSWER_RE.sub(
            lambda _m: f"<answer>{fixed}</answer>", comp, count=1)
    return True


def _repair_json(rows: List[Dict[str, Any]], num_ideas: int, tag: str) -> List[Dict[str, Any]]:
    """Run repair_answer_json over rows (in place) and say what it touched."""
    n = sum(1 for r in rows if repair_answer_json(r, num_ideas))
    print(f"[{tag}] JSON repair: rewrote {n}/{len(rows)} rows "
          f"(trailing commas before }} or ])")
    return rows


def _filter_valid_json(rows: List[Dict[str, Any]], num_ideas: int, tag: str) -> List[Dict[str, Any]]:
    """SFT: keep only rows whose <answer> parses to exactly `num_ideas` ideas."""
    keep = []
    n_bad_parse = n_bad_count = 0
    for r in rows:
        ans = _extract_answer(_completion(r))
        if ans is None:
            n_bad_parse += 1
            continue
        try:
            ideas = json.loads(ans)
        except Exception:
            n_bad_parse += 1
            continue
        if not isinstance(ideas, list) or len(ideas) != num_ideas:
            n_bad_count += 1
            continue
        keep.append(r)
    print(f"[{tag}] valid-JSON filter: kept {len(keep)} (dropped {n_bad_parse} unparseable, "
          f"{n_bad_count} != {num_ideas} ideas)")
    return keep


def _filter_seq_len(rows: List[Dict[str, Any]], num_ideas: int, max_seq_len: int,
                    tokenizer_name: str, tag: str,
                    prompt_type: str = DEFAULT_PROMPT_TYPE) -> List[Dict[str, Any]]:
    """SFT: drop rows whose tokenized (prompt + completion) exceeds max_seq_len,
    so the trainer's right-truncation never clips the <answer> close."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    def n_tok(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    keep = []
    over = []
    for r in rows:
        prompt, _ = _build_query(r, num_ideas, prompt_type)
        total = n_tok(prompt) + n_tok(_completion(r))
        if total <= max_seq_len:
            keep.append(r)
        else:
            over.append(total)
    dropped = len(over)
    pct = 100.0 * dropped / max(1, len(rows))
    extra = f" (dropped seq_len range {min(over)}-{max(over)})" if over else ""
    print(f"[{tag}] seq_len filter <= {max_seq_len}: kept {len(keep)} (dropped {dropped}, "
          f"{pct:.1f}%){extra}")
    return keep


def main() -> None:
    args = parse_args()
    if not args.in_path and not args.test_in_path:
        raise SystemExit("nothing to do: pass --in (train/val) and/or --test-in (test).")
    if args.stage == "sft" and args.test_in_path:
        raise SystemExit("--stage sft has no test split; drop --test-in (test is RL-schema only).")

    print(f"prompt_type={args.prompt_type} (answer format {_answer_fmt(args.prompt_type)}, "
          f"data_source {_data_source(args.prompt_type)}), num_ideas={args.num_ideas}")

    def emit(rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
        if args.stage == "rl":
            return [to_rl_record(r, split, i, args.num_ideas, args.prompt_type)
                    for i, r in enumerate(rows)]
        return [to_sft_record(r, split, i, args.num_ideas, args.messages_key,
                              args.prompt_type)
                for i, r in enumerate(rows)]

    # ---- train + val (sliced from --in) ----
    if args.in_path:
        rows = _read_rows(args.in_path, args.stream, "train-src")
        print(f"read {len(rows)} rows from {args.in_path}")
        usable = _usable(rows, args.min_documents, "train-src", args.prompt_type)

        # SFT-only quality/length gates (applied BEFORE the val slice so both
        # splits are clean and the holdout is representative of what we train on).
        if args.stage == "sft":
            # Repair BEFORE the filter, so a recoverable row is kept rather than counted
            # as a casualty. Repair itself is conservative -- see repair_answer_json().
            if args.repair_json:
                usable = _repair_json(usable, args.num_ideas, "train-src")
            if args.require_valid_json:
                usable = _filter_valid_json(usable, args.num_ideas, "train-src")
            if args.max_seq_len > 0:
                usable = _filter_seq_len(usable, args.num_ideas, args.max_seq_len,
                                         args.tokenizer, "train-src", args.prompt_type)

        n_val = args.n_val
        if n_val > len(usable):
            raise SystemExit(f"--n-val {n_val} exceeds usable rows {len(usable)}")
        val_rows = usable[len(usable) - n_val:] if n_val > 0 else []
        train_pool = usable[: len(usable) - n_val] if n_val > 0 else usable
        if args.n_train > 0:
            if args.n_train > len(train_pool):
                raise SystemExit(f"--n-train {args.n_train} exceeds available train rows {len(train_pool)}")
            train_rows = train_pool[: args.n_train]
        else:
            train_rows = train_pool

        _write(emit(train_rows, "train"), args.out_dir / "train.parquet")
        if val_rows:
            _write(emit(val_rows, "val"), args.out_dir / "val.parquet")

    # ---- test (whole of --test-in; no slicing; RL schema only) ----
    if args.test_in_path:
        test_rows = _read_rows(args.test_in_path, args.stream, "test-src")
        print(f"read {len(test_rows)} rows from {args.test_in_path}")
        test_usable = _usable(test_rows, args.min_documents, "test-src", args.prompt_type)
        _write(emit(test_usable, "test"), args.out_dir / "test.parquet")

    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
