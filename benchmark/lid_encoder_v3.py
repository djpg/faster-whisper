"""Calibrate adaptive LID on the same FLEURS rows; no extra inference per threshold.

Measure actual conservative and adaptive endpoints, then reconstruct policy quality
from paired predictions. Latency for intermediate thresholds is an estimate; it
includes measured legacy LID cost on fallback rows. Validate a selected threshold
on disjoint rows in a subsequent run, not on this calibration split.
"""
import argparse
import json
import time
from dataclasses import asdict
from unittest.mock import patch

from faster_whisper import WhisperModel
from faster_whisper.transcribe import _accept_language_reuse
from lid_encoder_fleurs_full import (
    MODELS, discover, emit, expected_language, plan,
    samples_for_locale, normalize, edits,
)

MARGINS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def infer(model, audio, margin, force_accept=False):
    with patch.object(model, "encode", wraps=model.encode) as encoder:
        legacy = model.detect_language

        def timed_legacy(*args, **kwargs):
            start = time.perf_counter()
            result = legacy(*args, **kwargs)
            legacy_times.append(time.perf_counter() - start)
            return result

        legacy_times = []
        with patch.object(model, "detect_language", side_effect=timed_legacy):
            with patch("faster_whisper.transcribe._accept_language_reuse",
                       wraps=(lambda p, m: True) if force_accept else _accept_language_reuse) as gate:
                start = time.perf_counter()
                segments, info = model.transcribe(
                    audio, beam_size=1, temperature=0.0, vad_filter=False,
                    language_reuse_margin=margin
                )
                segments = [asdict(s) for s in segments]
                elapsed = time.perf_counter() - start
                gated = gate.call_count > 0
        return segments, asdict(info), encoder.call_count, elapsed, sum(legacy_times), gated


def score(result, reference, expected, supported):
    segments, info, calls, elapsed, legacy_time, gated = result
    hyp = normalize(" ".join(s["text"] for s in segments))
    probs = sorted((p for _, p in info["all_language_probs"] or []), reverse=True)
    return {
        "word_errors": edits(reference.split(), hyp.split()),
        "char_errors": edits(reference.replace(" ", ""), hyp.replace(" ", "")),
        "language": info["language"],
        "lid_correct": int(info["language"] == expected) if supported else None,
        "probability": info["language_probability"],
        "margin": probs[0] - probs[1] if len(probs) >= 2 else None,
        "encoder_calls": calls, "latency_ms": elapsed * 1000,
        "legacy_lid_ms": legacy_time * 1000, "gated": gated,
        "text": hyp,
    }


def evaluate(args):
    revisions = json.loads(args.revisions)
    shards = discover()
    configs = args.locales.split(",")
    samples = {c: samples_for_locale(c, shards[c], args.samples, args.seed) for c in configs}
    for name in MODELS:
        model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=4,
                             revision=revisions[name])
        warmup = samples[configs[0]][0][1]
        infer(model, warmup, None)
        infer(model, warmup, 0., force_accept=True)
        for config in configs:
            expected, supported = expected_language(config, model)
            for i, (row, audio) in enumerate(samples[config]):
                if i % 2:
                    fast = infer(model, audio, 0., force_accept=True)
                    base = infer(model, audio, None)
                else:
                    base = infer(model, audio, None)
                    fast = infer(model, audio, 0., force_accept=True)
                ref = normalize(row["transcription"])
                a, b = score(base, ref, expected, supported), score(fast, ref, expected, supported)
                policies = []
                for margin in MARGINS:
                    accept = not b["gated"] or _accept_language_reuse(
                        fast[1]["all_language_probs"], margin
                    )
                    chosen = b if accept else a
                    policies.append({
                        "margin": margin, "fallback": not accept,
                        "word_errors": chosen["word_errors"],
                        "char_errors": chosen["char_errors"],
                        "lid_correct": chosen["lid_correct"],
                        "lid_changed": chosen["language"] != a["language"],
                        "text_changed": chosen["text"] != a["text"],
                        "encoder_calls_est": (b["encoder_calls"] if accept
                                              else a["encoder_calls"]),
                        # Fallback includes the extra fast LID decode; measure real
                        # end-to-end latency separately for the chosen policy.
                        "latency_ms_est": (b["latency_ms"] if accept else
                                           a["latency_ms"]),
                    })
                a.pop("text")
                b.pop("text")
                emit({"kind": "v3_calibration_sample", "model": name, "config": config,
                      "source_revision": row["source_revision"],
                      "source_shard": row["source_shard"], "source_row": row["source_row"],
                      "id": row["id"], "duration_s": len(audio) / 16000,
                      "reference_words": len(ref.split()),
                      "reference_chars": len(ref.replace(" ", "")),
                      "supported": supported, "v1": a, "fast": b, "policies": policies})
        del model
    emit({"kind": "complete", "configs": configs,
          "pairs": sum(map(len, samples.values())) * len(MODELS)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--locales")
    parser.add_argument("--revisions")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    plan() if args.plan else evaluate(args)
