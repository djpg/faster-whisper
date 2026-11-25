"""V1 versus V2: short audio latency and multilingual FLEURS quality."""

import argparse
import io
import json
import statistics
import unicodedata

import numpy as np

from faster_whisper import WhisperModel, decode_audio
from lid_encoder_reuse import load_baseline, run, summarize


def emit(row):
    print(json.dumps(row, ensure_ascii=False), flush=True)


def speed(args):
    baseline_cls = load_baseline(args.baseline)
    source = decode_audio("tests/data/jfk.flac")
    for name in args.models:
        kwargs = dict(device="cpu", compute_type="int8", cpu_threads=4)
        original, optimized = baseline_cls(name, **kwargs), WhisperModel(name, **kwargs)
        for duration in (1, 3, 5, 10, 29, 30):
            audio = np.resize(source, duration * 16000)
            for beam in (1, 5):
                for _ in range(2):
                    run(original, audio, beam)
                    run(optimized, audio, beam)
                before, after = [], []
                for i in range(args.repeats):
                    if i % 2:
                        after.append(run(optimized, audio, beam))
                        before.append(run(original, audio, beam))
                    else:
                        before.append(run(original, audio, beam))
                        after.append(run(optimized, audio, beam))
                a, b = summarize(before, duration), summarize(after, duration)
                row = {
                    "kind": "speed",
                    "model": name,
                    "duration_s": duration,
                    "beam_size": beam,
                    "v1": a,
                    "v2": b,
                    "speedup": round(a["p50_ms"] / b["p50_ms"], 3),
                    "latency_reduction_pct": round(
                        100 * (1 - b["p50_ms"] / a["p50_ms"]), 2
                    ),
                    "same_segments": sum(x[2] == y[2] for x, y in zip(before, after)),
                    "same_language": sum(
                        x[3]["language"] == y[3]["language"]
                        for x, y in zip(before, after)
                    ),
                    "max_language_probability_delta": max(
                        abs(x[3]["language_probability"] - y[3]["language_probability"])
                        for x, y in zip(before, after)
                    ),
                }
                emit(row)
                if duration < 30:
                    assert all(x[1] - y[1] == 1 for x, y in zip(before, after))
                else:
                    assert all(x[2:] == y[2:] for x, y in zip(before, after))
        del original, optimized


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def edits(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def load_fleurs(count):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info("google/fleurs").sha
    emit({"kind": "dataset", "dataset": "google/fleurs", "revision": revision,
          "split": "test", "per_language": count})
    samples = []
    for config, language in (
        ("en_us", "en"), ("es_419", "es"), ("fr_fr", "fr"),
        ("de_de", "de"), ("ar_eg", "ar"),
    ):
        path = hf_hub_download(
            "google/fleurs", f"{config}/test-00000-of-00001.parquet",
            repo_type="dataset", revision=revision
        )
        selected = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=16, columns=["id", "audio", "transcription"]
        ):
            for item in batch.to_pylist():
                audio = decode_audio(io.BytesIO(item["audio"]["bytes"]))
                if not 0 < len(audio) < 30 * 16000 or not normalize(item["transcription"]):
                    continue
                samples.append((config, language, item["id"], audio, item["transcription"]))
                selected += 1
                if selected == count:
                    break
            if selected == count:
                break
        if selected != count:
            raise RuntimeError(f"Insufficient eligible samples: {config}: {selected}")
        emit({"kind": "samples", "config": config, "count": selected})
    return samples


def quality(args):
    samples = load_fleurs(args.samples)
    baseline_cls = load_baseline(args.baseline)
    for name in args.models:
        kwargs = dict(device="cpu", compute_type="int8", cpu_threads=4)
        original, optimized = baseline_cls(name, **kwargs), WhisperModel(name, **kwargs)
        totals = {}
        for i, (config, lang, sample_id, audio, reference) in enumerate(samples):
            if i % 2:
                b, a = run(optimized, audio, 1), run(original, audio, 1)
            else:
                a, b = run(original, audio, 1), run(optimized, audio, 1)
            ref = normalize(reference)
            hypotheses = [normalize(" ".join(s["text"] for s in x[2])) for x in (a, b)]
            words = ref.split()
            chars = ref.replace(" ", "")
            row = {
                "kind": "quality_sample", "model": name, "config": config,
                "id": sample_id, "duration_s": len(audio) / 16000,
                "reference_words": len(words), "reference_chars": len(chars),
                "word_errors": [edits(words, h.split()) for h in hypotheses],
                "char_errors": [edits(chars, h.replace(" ", "")) for h in hypotheses],
                "lid_correct": [int(x[3]["language"] == lang) for x in (a, b)],
                "lid_changed": int(a[3]["language"] != b[3]["language"]),
                "probability_delta": abs(a[3]["language_probability"] -
                                         b[3]["language_probability"]),
                "text_changed": int(hypotheses[0] != hypotheses[1]),
                "segments_changed": int(a[2] != b[2]),
                "encoder_calls": [a[1], b[1]],
                "latency_ms": [a[0] * 1000, b[0] * 1000],
            }
            emit(row)
            for key in (config, "ALL"):
                totals.setdefault(key, []).append(row)
        for config, rows in totals.items():
            n = len(rows)
            total_words = sum(r["reference_words"] for r in rows)
            total_chars = sum(r["reference_chars"] for r in rows)
            emit({
                "kind": "quality_summary", "model": name, "config": config,
                "n": n,
                "wer_pct": [100 * sum(r["word_errors"][k] for r in rows) / total_words
                            for k in (0, 1)],
                "cer_pct": [100 * sum(r["char_errors"][k] for r in rows) / total_chars
                            for k in (0, 1)],
                "lid_accuracy_pct": [100 * sum(r["lid_correct"][k] for r in rows) / n
                                     for k in (0, 1)],
                "lid_changed": sum(r["lid_changed"] for r in rows),
                "text_changed": sum(r["text_changed"] for r in rows),
                "segments_changed": sum(r["segments_changed"] for r in rows),
                "mean_probability_delta": statistics.mean(
                    r["probability_delta"] for r in rows
                ),
                "max_probability_delta": max(r["probability_delta"] for r in rows),
                "mean_encoder_calls": [statistics.mean(r["encoder_calls"][k] for r in rows)
                                       for k in (0, 1)],
                "median_latency_ms": [statistics.median(r["latency_ms"][k] for r in rows)
                                      for k in (0, 1)],
            })
        del original, optimized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--mode", choices=["speed", "quality"], required=True)
    parser.add_argument("--models", nargs="+", default=["tiny", "small"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    speed(args) if args.mode == "speed" else quality(args)
