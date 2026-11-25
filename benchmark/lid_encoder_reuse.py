"""Compare the original and optimized WhisperModel on the same CPU runner."""

import argparse
import json
import statistics
import subprocess
import sys
import time
import types

from dataclasses import asdict
from unittest.mock import patch

import numpy as np

from faster_whisper import WhisperModel, decode_audio


def load_baseline(ref):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:faster_whisper/transcribe.py"], text=True
    )
    module = types.ModuleType("faster_whisper._benchmark_baseline")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{ref}:transcribe.py", "exec"), module.__dict__)
    return module.WhisperModel


def run(model, audio, beam_size):
    with patch.object(model, "encode", wraps=model.encode) as encode:
        start = time.perf_counter()
        segments, info = model.transcribe(
            audio, beam_size=beam_size, temperature=0.0, vad_filter=False
        )
        segments = list(segments)
        elapsed = time.perf_counter() - start
        count = encode.call_count
    return elapsed, count, [asdict(s) for s in segments], asdict(info)


def summarize(results, duration):
    times = [row[0] for row in results]
    return {
        "p50_ms": round(statistics.median(times) * 1000, 2),
        "p95_ms": round(float(np.percentile(times, 95)) * 1000, 2),
        "rtf": round(statistics.median(times) / duration, 5),
        "encoder_calls": [row[1] for row in results],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--audio", default="tests/data/jfk.flac")
    parser.add_argument("--models", nargs="+", default=["tiny", "small"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    baseline_cls = load_baseline(args.baseline)
    source = decode_audio(args.audio)
    print("Configuration:", vars(args), flush=True)
    for name in args.models:
        kwargs = dict(device="cpu", compute_type="int8", cpu_threads=4)
        original = baseline_cls(name, **kwargs)
        optimized = WhisperModel(name, **kwargs)
        for duration in (1, 3, 5, 10, 30, 60):
            audio = np.resize(source, duration * 16000)
            for beam in (1, 5):
                for _ in range(args.warmup):
                    run(original, audio, beam)
                    run(optimized, audio, beam)
                before, after = [], []
                for i in range(args.repeats):
                    if i % 2 == 0:
                        before.append(run(original, audio, beam))
                        after.append(run(optimized, audio, beam))
                    else:
                        after.append(run(optimized, audio, beam))
                        before.append(run(original, audio, beam))
                a = summarize(before, duration)
                b = summarize(after, duration)
                equal = all(x[2:] == y[2:] for x, y in zip(before, after))
                row = {
                    "model": name,
                    "duration_s": duration,
                    "beam_size": beam,
                    "baseline": a,
                    "optimized": b,
                    "speedup": round(a["p50_ms"] / b["p50_ms"], 3),
                    "latency_reduction_pct": round(
                        100 * (1 - b["p50_ms"] / a["p50_ms"]), 2
                    ),
                    "identical_segments_and_info": equal,
                }
                print(json.dumps(row), flush=True)
                if not equal:
                    raise AssertionError("Transcription or metadata changed")
        del original, optimized


if __name__ == "__main__":
    main()
