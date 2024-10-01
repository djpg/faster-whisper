
export MODEL_PATH="large-v3"
export LANG="es"
export COMPUTE_TYPE="float16"

python3 speed_benchmark.py
python3 memory_benchmark.py
python3 wer_benchmark.py
