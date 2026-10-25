#!/usr/bin/env bash
# Builds the browser engine (docs/wasm/tickforge.js + .wasm) with Emscripten.
# Needs emcc on PATH (https://emscripten.org/docs/getting_started/downloads.html).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p docs/wasm
emcc -O3 -std=c++20 -Iinclude -DNDEBUG \
  web/wasm_api.cpp src/synth.cpp src/backtest.cpp src/tick_store.cpp \
  -sMODULARIZE=1 -sEXPORT_NAME=createTickforge -sENVIRONMENT=worker,node \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB -sMAXIMUM_MEMORY=1GB \
  -sEXPORTED_RUNTIME_METHODS=HEAPF64,HEAP8,HEAP32 \
  -o docs/wasm/tickforge.js
ls -l docs/wasm
