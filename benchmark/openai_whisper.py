# pip install -U openai-whisper
import os
import time
import logging

import whisper


model_path = os.getenv("MODEL_PATH", "large-v3")
lang = os.getenv("LANG", "en")
audio_file = "benchmark.m4a" if lang == "fr" else f"benchmark_{lang}.m4a"
compute_type = os.getenv("COMPUTE_TYPE", "default")
fp16 = True if compute_type in ["float16", "default"] else False
logging.info(f"model_path: {model_path} lang: {lang} compute_type: {compute_type}")

model = whisper.load_model(model_path)
init_time = time.time()
result = model.transcribe(audio_file, beam_size=5, language=lang, fp16=fp16, verbose=False)
end_time = time.time()
exec_time = round(end_time - init_time, 3)
print(result["text"])
print(f"Execution time: {exec_time}")
