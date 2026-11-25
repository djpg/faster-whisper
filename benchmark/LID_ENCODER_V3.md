# Adaptive LID encoder reuse (V3)

`WhisperModel.transcribe(..., language_reuse_margin=0.30)` opts into
adaptive reuse on short clips. The default `None` preserves V1 behavior.
The threshold compares the two highest probabilities from the ASR feature
window, not the legacy LID input. When the margin is insufficient, legacy
LID is rerun and the already-computed ASR encoder is retained. Confidence
metadata on accepted clips can differ from legacy. This is not a guarantee
of identical labels. Explicit languages and custom clips remain unchanged.
BatchedInferencePipeline is outside this change.

The calibration workflow compares V1 and the always-accept endpoint on the
same pinned FLEURS samples as V2. It computes policy outcomes for eight
thresholds without re-transcribing each sample eight times. Estimated
latency for fallback uses V1 latency as a proxy and omits extra LID decode
overhead. The selected policy must be tested with actual inference and
disjoint samples before recommending a default.
