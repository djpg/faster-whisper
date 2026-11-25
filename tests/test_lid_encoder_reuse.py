from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

from faster_whisper import WhisperModel, decode_audio
from faster_whisper.audio import pad_or_trim


@pytest.fixture(scope="module")
def model():
    return WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)


@pytest.mark.parametrize(
    "duration,clip_timestamps,detection_segments,threshold,expected_saved",
    [
        (1, "0", 1, 0.5, 1),
        (5, "0", 1, 0.5, 1),
        (10, "0", 2, 1.0, 1),
        (29, "0", 1, 0.5, 1),
        (30, "0", 1, 0.5, 1),
        (60, "0", 2, 1.0, 1),
        (30, "0,5", 1, 0.5, 0),
        (30, "1", 1, 0.5, 0),
    ],
)
def test_reuse_lid_encoder(
    model, jfk_path, duration, clip_timestamps, detection_segments, threshold,
    expected_saved,
):
    source = decode_audio(jfk_path)
    audio = np.resize(source, duration * 16000)
    options = dict(
        beam_size=1,
        temperature=0.0,
        vad_filter=False,
        clip_timestamps=clip_timestamps,
        language_detection_segments=detection_segments,
        language_detection_threshold=threshold,
        word_timestamps=True,
    )
    detect = model._detect_language_from_features

    def without_reuse(*args, **kwargs):
        kwargs["reuse_first_encoder"] = False
        return detect(*args, **kwargs)

    def run():
        with patch.object(model, "encode", wraps=model.encode) as encode:
            segments, info = model.transcribe(audio, **options)
            segments = [asdict(segment) for segment in segments]
            return segments, asdict(info), encode.call_count

    with patch.object(
        model, "_detect_language_from_features", side_effect=without_reuse
    ):
        baseline = run()
    optimized = run()

    assert baseline[:2] == optimized[:2]
    assert baseline[2] - optimized[2] == expected_saved


def test_detect_language_public_api(model, jfk_path):
    result = model.detect_language(audio=decode_audio(jfk_path))
    assert len(result) == 3
    assert result[0] == "en"


@pytest.mark.parametrize("duration", [1, 5, 10, 29])
def test_short_lid_asr_same_features(model, jfk_path, duration):
    audio = np.resize(decode_audio(jfk_path), duration * 16000)
    original_encode = model.encode
    inputs = []

    def capture(features):
        inputs.append(features.copy())
        return original_encode(features)

    with patch.object(model, "encode", side_effect=capture):
        segments, info = model.transcribe(
            audio, beam_size=1, temperature=0.0, without_timestamps=True
        )
        list(segments)

    assert len(inputs) == 1
    expected = pad_or_trim(model.feature_extractor(audio)[..., :-1])
    np.testing.assert_array_equal(inputs[0], expected)
    assert info.language in model.supported_languages


def test_known_language_and_empty_audio(model, jfk_path):
    audio = decode_audio(jfk_path)[: 5 * 16000]
    with patch.object(model, "encode", wraps=model.encode) as encode:
        list(model.transcribe(audio, language="en", without_timestamps=True)[0])
        assert encode.call_count == 1
    assert list(model.transcribe(np.array([], dtype=np.float32))[0]) == []
