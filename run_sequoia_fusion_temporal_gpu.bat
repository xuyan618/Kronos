@echo off
REM ============================================================
REM Kronos fusion model training + backtest - native Windows (RTX 3060 GPU)
REM Usage: place repo at C:\Kronos, double-click this file; or in cmd: cd C:\Kronos then run
REM        run_sequoia_fusion_temporal_gpu.bat
REM Prereqs: Python 3.12 on PATH; torch(cu121) + requirements.txt installed
REM        sequoia_v2.db copied to C:\Kronos\data\sequoia_v2.db
REM        Kronos pretrained weights auto-download from HuggingFace on first run
REM ============================================================

cd /d %~dp0

REM ---- Path (edit as needed) ----
set "SEQUOIA_DB=C:\Kronos\data\sequoia_v2.db"

REM ---- HuggingFace: download online (first run pulls NeoQuasar/Kronos-base and Tokenizer) ----
set "HF_HOME=C:\Kronos\.hf_cache"
REM Use mirror if huggingface.co is slow/blocked in your region; comment out next line if direct works
set "HF_ENDPOINT=https://hf-mirror.com"
REM Uncomment next line if you pre-copied weights into HF_HOME and want fully offline
REM set "HF_HUB_OFFLINE=1"

REM ---- Data / text encoding ----
set "SEQUOIA_FUSION=1"
set "SEQUOIA=1"
set "SEQUOIA_CPU_DATASET_PATH=.\data\sequoia_fusion_temporal"
set "SEQUOIA_CPU_MAX_SYMBOLS=300"
REM hash = offline deterministic hash fallback (no network, no transformers); auto = prefer FinBERT, degrade on fail
set "TEXT_ENCODER_MODE=hash"

REM ---- Training hyperparams (GPU scaled; 300 symbols for minimal dataset) ----
set "FUSION_TRAIN_ITER=5000"
set "FUSION_VAL_ITER=200"
set "FUSION_BATCH_SIZE=64"
set "FUSION_EPOCHS=10"
set "FUSION_NUM_WORKERS=4"
set "SEQUOIA_DETERMINISTIC=0"

REM ---- Time splits (strict chronological, all three segments have text) ----
set "TRAIN_RANGE_START=2024-01-02"
set "TRAIN_RANGE_END=2025-09-30"
set "VAL_RANGE_START=2025-10-01"
set "VAL_RANGE_END=2026-03-31"
set "TEST_RANGE_START=2026-04-01"
set "TEST_RANGE_END=2026-09-14"
set "TRAIN_MIN_LEN=101"
set "VAL_MIN_LEN=0"
set "TEST_MIN_LEN=0"

set "PYTHONPATH=%~dp0;%PYTHONPATH%"

echo == 1. Generate price data ==
python -u finetune\sequoia_db_loader.py || goto :error

echo == 2. Rebuild full-history text embeddings ==
python -u finetune\build_text_embeddings.py || goto :error

REM ---- Windows: torchrun's elastic rendezvous needs libuv, absent in the official Windows wheel.
REM ---- For single-GPU (nproc_per_node=1) we skip torchrun and launch python directly with the
REM ---- distributed env vars torchrun would have set, and disable libuv for the c10d store. ----
set "USE_LIBUV=0"
set "MASTER_ADDR=localhost"
set "MASTER_PORT=29500"
set "RANK=0"
set "WORLD_SIZE=1"
set "LOCAL_RANK=0"

echo == 3. Fine-tune KronosFusion (GPU / single process) ==
python -u finetune\train_predictor_fusion.py || goto :error

echo == 4. Deterministic backtest (OOS, CPU) ==
set "MAX_WINDOW=250"
python -u finetune\backtest_predictor_fusion.py || goto :error

echo Task complete. OOS results in .\outputs\models\sequoia_predictor_fusion\backtest\
goto :eof

:error
echo [ERROR] Step failed, exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%
