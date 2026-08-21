#!/bin/bash
# ============================================================================
# BPE Tokenizer Benchmark: Python vs C++
#
# 用法:
#   chmod +x run_benchmark.sh
#   ./run_benchmark.sh <input_file> <vocab_size>
#
# 例:
#   ./run_benchmark.sh ./toy.txt 1000
# ============================================================================

set -e

INPUT_FILE=${1:?"Usage: $0 <input_file> <vocab_size>"}
VOCAB_SIZE=${2:?"Usage: $0 <input_file> <vocab_size>"}
NUM_PYTHON_PROCS=4

echo "=================================================================="
echo "  BPE Tokenizer Benchmark: Python vs C++"
echo "=================================================================="
echo "  Input:      $INPUT_FILE"
echo "  Vocab size: $VOCAB_SIZE"
echo "  Date:       $(date)"
echo "=================================================================="

# --- Step 1: Compile C++ (if needed) ---
if [ ! -f ./bpe_benchmark ]; then
    echo ""
    echo "[Compiling C++...]"
    g++ -std=c++20 -O2 -pthread \
        bpe_tokenizer.cpp pretokenizer.cpp parallel_helper.cpp main_benchmark.cpp \
        -o bpe_benchmark
    echo "  Done."
fi

# --- Step 2: Run C++ benchmark ---
echo ""
echo "============ C++ Benchmark ============"
./bpe_benchmark "$INPUT_FILE" "$VOCAB_SIZE" 2>&1 | tee /tmp/cpp_benchmark_output.txt

# 从结果表中提取时间 — 只匹配表格行 (以 ║ 开头)
# 表格行格式: ║ Serial (1 thread)        ║     8.451s ║ ...
CPP_SERIAL=$(grep "║ Serial" /tmp/cpp_benchmark_output.txt | grep -oE '[0-9]+\.[0-9]+s' | tr -d 's')
CPP_PARALLEL=$(grep "║ Parallel" /tmp/cpp_benchmark_output.txt | grep -oE '[0-9]+\.[0-9]+s' | tr -d 's')

echo ""
echo "[DEBUG] C++ serial time:   ${CPP_SERIAL}s"
echo "[DEBUG] C++ parallel time: ${CPP_PARALLEL}s"

# --- Step 3: Run Python benchmark ---
echo ""
echo "========== Python Benchmark ==========="
python benchmark_python.py "$INPUT_FILE" "$VOCAB_SIZE" "$NUM_PYTHON_PROCS" 2>&1 | tee /tmp/python_benchmark_output.txt

# 从结果表中提取时间
PY_SERIAL=$(grep "║ Python (1 proc)" /tmp/python_benchmark_output.txt | grep -oE '[0-9]+\.[0-9]+s' | tr -d 's')
PY_PARALLEL=$(grep "║ Python ($NUM_PYTHON_PROCS proc)" /tmp/python_benchmark_output.txt | grep -oE '[0-9]+\.[0-9]+s' | tr -d 's')

echo ""
echo "[DEBUG] Python serial time:   ${PY_SERIAL}s"
echo "[DEBUG] Python parallel time: ${PY_PARALLEL}s"

# --- Step 4: Summary ---
echo ""
echo "=================================================================="
echo "  FINAL COMPARISON"
echo "=================================================================="
echo ""
printf "%-30s %12s\n" "Configuration" "Time (s)"
echo "----------------------------------------------"
printf "%-30s %12s\n" "Python serial (1 proc)" "${PY_SERIAL:-N/A}s"
printf "%-30s %12s\n" "Python parallel (${NUM_PYTHON_PROCS} procs)" "${PY_PARALLEL:-N/A}s"
printf "%-30s %12s\n" "C++ serial (1 thread)" "${CPP_SERIAL:-N/A}s"
printf "%-30s %12s\n" "C++ parallel (4 threads)" "${CPP_PARALLEL:-N/A}s"
echo ""

# 计算 speedup
if [ -n "$PY_SERIAL" ] && [ -n "$CPP_SERIAL" ]; then
    SPEEDUP=$(echo "$PY_SERIAL $CPP_SERIAL" | awk '{printf "%.2f", $1/$2}')
    echo "  C++ serial   vs Python serial:   ${SPEEDUP}x"
fi

if [ -n "$PY_PARALLEL" ] && [ -n "$CPP_PARALLEL" ]; then
    SPEEDUP=$(echo "$PY_PARALLEL $CPP_PARALLEL" | awk '{printf "%.2f", $1/$2}')
    echo "  C++ parallel vs Python parallel: ${SPEEDUP}x"
fi

if [ -n "$PY_SERIAL" ] && [ -n "$CPP_PARALLEL" ]; then
    SPEEDUP=$(echo "$PY_SERIAL $CPP_PARALLEL" | awk '{printf "%.2f", $1/$2}')
    echo "  C++ parallel vs Python serial:   ${SPEEDUP}x  ← best case"
fi

if [ -n "$CPP_SERIAL" ] && [ -n "$CPP_PARALLEL" ]; then
    SPEEDUP=$(echo "$CPP_SERIAL $CPP_PARALLEL" | awk '{printf "%.2f", $1/$2}')
    echo "  C++ parallel vs C++ serial:      ${SPEEDUP}x  ← threading gain"
fi

echo ""
echo "=================================================================="
