import os
import logging

from threading import Thread
from typing import Optional

from faster_whisper import WhisperModel

AUDIOS = {"fr": "benchmark.m4a", "en": "benchmark_en.m4a"}
model_path = os.getenv("MODEL_PATH", "large-v3")
lang = os.getenv("LANG", "fr")
compute_type = os.getenv("COMPUTE_TYPE", "default")
model = WhisperModel(model_path, device="cuda", compute_type=compute_type)
logging.info(f"model_path: {model_path} lang: {lang} compute_type: {compute_type}")


def inference(print_output: bool=True):
    audio_file = AUDIOS[lang]
    segments, info = model.transcribe(audio_file, language=lang)
    if print_output:
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    else:
        _ = list(segments)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class MyThread(Thread):
    def __init__(self, func, params):
        super(MyThread, self).__init__()
        self.func = func
        self.params = params
        self.result = None

    def run(self):
        self.result = self.func(*self.params)

    def get_result(self):
        return self.result
