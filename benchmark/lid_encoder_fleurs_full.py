"""Paired FLEURS test evaluation of conservative V1 vs short-audio V2.

All 102 locales are evaluated for ASR; unsupported LID labels are excluded
from supported-language accuracy, not silently treated as correct aliases.
JSONL logs are the durable benchmark output (no uploaded artifacts).
"""

import argparse
import io
import json
import os
import random
import re
import statistics

from huggingface_hub import HfApi, hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq

from faster_whisper import WhisperModel, decode_audio
from lid_encoder_reuse import load_baseline, run
from lid_encoder_reuse_v2 import normalize, edits

DATASET = "google/fleurs"
REVISION = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"
MODELS = {"tiny": "Systran/faster-whisper-tiny",
          "small": "Systran/faster-whisper-small"}
ALIASES = {"cmn": "zh", "fil": "tl", "jv": "jw", "nb": "no"}
CER_PRIMARY = {"cmn_hans_cn", "yue_hant_hk", "ja_jp", "th_th",
               "lo_la", "my_mm", "km_kh", "ko_kr", "vi_vn"}


def emit(row):
    print(json.dumps(row, ensure_ascii=False), flush=True)


def discover():
    api = HfApi()
    files = api.list_repo_files(DATASET, repo_type="dataset", revision=REVISION)
    shards = {}
    for path in files:
        parts = path.split("/")
        if len(parts) < 2 or not path.endswith(".parquet") or parts[0] == "all":
            continue
        # Converted datasets can use either config/test/0000.parquet
        # or config/test-00000-of-00001.parquet.
        if ((len(parts) >= 3 and parts[1] == "test")
                or (len(parts) == 2 and parts[1].startswith("test-"))):
            shards.setdefault(parts[0], []).append((REVISION, path))

    # The pinned conversion snapshot has 101 discovered locales. Use a
    # separately pinned, published Afrikaans test shard if it is absent.
    if "af_za" not in shards:
        af_revision = "8e74c3a7c16ffd851ceb9d50012daa64122d6734"
        af_files = api.list_repo_files(DATASET, repo_type="dataset", revision=af_revision)
        af_paths = sorted(p for p in af_files
                          if p.startswith("af_za/test-") and p.endswith(".parquet"))
        if not af_paths:
            raise RuntimeError(f"No Afrikaans test files at {af_revision}")
        shards["af_za"] = [(af_revision, p) for p in af_paths]

    if len(shards) != 102:
        raise RuntimeError(f"Expected 102 test locales, got {len(shards)}: {sorted(shards)}")
    return {k: sorted(v) for k, v in sorted(shards.items())}


def plan():
    shards = discover()
    locales = list(shards)
    # Keep model weights identical to the original full benchmark run.
    revisions = {"tiny": "d90ca5fe260221311c53c58e660288d3deb8d356",
                 "small": "536b0662742c02347bc0e980a01041f333bce120"}
    matrix = {"include": [
        {"shard": i // 3 + 1, "locales": ",".join(locales[i:i + 3])}
        for i in range(0, len(locales), 3)
    ]}
    requested = os.environ.get("FLEURS_SHARDS", "").strip()
    if not requested:
        match = re.search(r"\[shards:([0-9, ]+)\]",
                          os.environ.get("FLEURS_COMMIT_MESSAGE", ""))
        if match:
            requested = match.group(1)
    if requested:
        selected = {int(x.strip()) for x in requested.split(",")}
        available = {entry["shard"] for entry in matrix["include"]}
        if not selected or not selected <= available:
            raise ValueError(f"Invalid shard IDs: {selected - available}")
        matrix["include"] = [entry for entry in matrix["include"]
                             if entry["shard"] in selected]
    emit({"kind": "plan", "dataset_revision": REVISION, "locales": locales,
          "model_revisions": revisions, "matrix": matrix,
          "af_za_sources": shards["af_za"]})
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write("matrix=" + json.dumps(matrix, separators=(",", ":")) + "\n")
        f.write("revisions=" + json.dumps(revisions, separators=(",", ":")) + "\n")


def samples_for_locale(config, paths, count, seed):
    tables = []
    for revision, shard in paths:
        path = hf_hub_download(DATASET, shard, repo_type="dataset", revision=revision)
        table = pq.read_table(path, columns=["id", "audio", "transcription"])
        # Preserve the source shard and row index for reproducibility.
        table = table.append_column("source_shard", pa.array([shard] * len(table)))
        table = table.append_column("source_revision", pa.array([revision] * len(table)))
        table = table.append_column("source_row", pa.array(range(len(table))))
        tables.append(table)
    table = pa.concat_tables(tables)
    indices = list(range(len(table)))
    random.Random(f"{seed}:{config}").shuffle(indices)
    samples = []
    for i in indices:
        row = table.slice(i, 1).to_pylist()[0]
        if not normalize(row["transcription"]):
            continue
        audio = decode_audio(io.BytesIO(row["audio"]["bytes"]))
        if not 0 < len(audio) < 30 * 16000:
            continue
        samples.append((row, audio))
        if len(samples) == count:
            break
    emit({"kind": "selection", "config": config, "seed": seed,
          "available_rows": len(table), "selected": len(samples), "requested": count})
    if not samples:
        raise RuntimeError(f"{config}: no eligible test samples")
    if len(samples) < count:
        emit({"kind": "sample_shortfall", "config": config,
              "selected": len(samples), "requested": count,
              "missing": count - len(samples),
              "policy": "all eligible rows, no replacement"})
    return samples


def expected_language(config, model):
    code = config.split("_")[0]
    code = ALIASES.get(code, code)
    supported = (code in model.supported_languages
                 and model.hf_tokenizer.token_to_id(f"<|{code}|>") is not None)
    return code, supported


def score_pair(original, optimized, config, row, audio, index, expected, supported):
    if index % 2:
        b, a = run(optimized, audio, 1), run(original, audio, 1)
    else:
        a, b = run(original, audio, 1), run(optimized, audio, 1)
    ref = normalize(row["transcription"])
    words, chars = ref.split(), ref.replace(" ", "")
    raw = [" ".join(s["text"] for s in r[2]) for r in (a, b)]
    hypotheses = [normalize(t) for t in raw]
    languages = [r[3]["language"] for r in (a, b)]
    probabilities = [r[3]["language_probability"] for r in (a, b)]
    old_probs = sorted(a[3]["all_language_probs"] or [], key=lambda p: p[1], reverse=True)
    margin = old_probs[0][1] - old_probs[1][1] if len(old_probs) > 1 else None
    return {
        "id": row["id"], "source_shard": row["source_shard"],
        "source_row": row["source_row"], "source_revision": row["source_revision"],
        "duration_s": len(audio) / 16000,
        "reference_words": len(words), "reference_chars": len(chars),
        "word_errors": [edits(words, h.split()) for h in hypotheses],
        "char_errors": [edits(chars, h.replace(" ", "")) for h in hypotheses],
        "languages": languages, "probabilities": probabilities,
        "lid_correct": [int(l == expected) for l in languages] if supported else None,
        "lid_changed": int(languages[0] != languages[1]),
        "v1_lid_margin": margin,
        "probability_delta": abs(probabilities[0] - probabilities[1]),
        "text_changed": int(hypotheses[0] != hypotheses[1]),
        "raw_text_changed": int(raw[0] != raw[1]),
        "segments_changed": int(a[2] != b[2]),
        "encoder_calls": [a[1], b[1]],
        "latency_ms": [round(a[0] * 1000, 3), round(b[0] * 1000, 3)],
    }


def summary(name, config, expected, supported, rows):
    n = len(rows)
    sums = lambda key: [sum(r[key][k] for r in rows) for k in (0, 1)]
    words = sum(r["reference_words"] for r in rows)
    chars = sum(r["reference_chars"] for r in rows)
    word_errors, char_errors = sums("word_errors"), sums("char_errors")
    medians = [statistics.median(r["latency_ms"][k] for r in rows) for k in (0, 1)]
    lid = sums("lid_correct") if supported else None
    emit({
        "kind": "fleurs_summary", "model": name, "config": config, "n": n,
        "expected_language": expected, "lid_supported": supported,
        "primary_metric": "CER" if config in CER_PRIMARY else "WER",
        "reference_words": words, "reference_chars": chars,
        "word_errors": word_errors, "char_errors": char_errors,
        "wer_pct": [100 * e / words for e in word_errors],
        "cer_pct": [100 * e / chars for e in char_errors],
        "lid_correct": lid, "lid_accuracy_pct": [100 * x / n for x in lid] if lid else None,
        "lid_changed": sum(r["lid_changed"] for r in rows),
        "lid_regressions": sum(r["lid_correct"] == [1, 0] for r in rows) if supported else None,
        "lid_improvements": sum(r["lid_correct"] == [0, 1] for r in rows) if supported else None,
        "word_error_regressions": sum(r["word_errors"][1] > r["word_errors"][0] for r in rows),
        "word_error_improvements": sum(r["word_errors"][1] < r["word_errors"][0] for r in rows),
        "char_error_regressions": sum(r["char_errors"][1] > r["char_errors"][0] for r in rows),
        "char_error_improvements": sum(r["char_errors"][1] < r["char_errors"][0] for r in rows),
        "text_changed": sum(r["text_changed"] for r in rows),
        "raw_text_changed": sum(r["raw_text_changed"] for r in rows),
        "segments_changed": sum(r["segments_changed"] for r in rows),
        "sum_probability_delta": sum(r["probability_delta"] for r in rows),
        "max_probability_delta": max(r["probability_delta"] for r in rows),
        "encoder_calls": sums("encoder_calls"),
        "one_encoder_saved": sum(r["encoder_calls"][0] - r["encoder_calls"][1] == 1 for r in rows),
        "latency_sum_ms": sums("latency_ms"), "median_latency_ms": medians,
        "median_latency_reduction_pct": 100 * (1 - medians[1] / medians[0]),
    })


def evaluate(args):
    revisions = json.loads(args.revisions)
    shards = discover()
    configs = args.locales.split(",")
    samples = {c: samples_for_locale(c, shards[c], args.samples, args.seed) for c in configs}
    baseline = load_baseline(args.baseline)
    emit({"kind": "configuration", "baseline": args.baseline, "dataset_revision": REVISION,
          "model_revisions": revisions, "configs": configs, "samples": args.samples,
          "seed": args.seed, "beam_size": 1, "compute_type": "int8", "cpu_threads": 4})
    for name in MODELS:
        options = dict(device="cpu", compute_type="int8", cpu_threads=4,
                       revision=revisions[name])
        original, optimized = baseline(name, **options), WhisperModel(name, **options)
        warmup_audio = samples[configs[0]][0][1]
        run(original, warmup_audio, 1)
        run(optimized, warmup_audio, 1)
        for config in configs:
            expected, supported = expected_language(config, optimized)
            rows = []
            for i, (row, audio) in enumerate(samples[config]):
                result = score_pair(original, optimized, config, row, audio, i, expected, supported)
                rows.append(result)
                emit({"kind": "fleurs_sample", "model": name, "config": config, **result})
            summary(name, config, expected, supported, rows)
        del original, optimized
    actual = {config: len(samples[config]) for config in configs}
    emit({"kind": "complete", "configs": configs, "samples_by_config": actual,
          "requested_per_config": args.samples,
          "unique_rows": sum(actual.values()),
          "pairs": sum(actual.values()) * len(MODELS),
          "transcriptions": sum(actual.values()) * len(MODELS) * 2})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--baseline", default="fd4501da08f28840b0306e734a3374c9645345e1")
    parser.add_argument("--locales")
    parser.add_argument("--revisions")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    plan() if args.plan else evaluate(args)
