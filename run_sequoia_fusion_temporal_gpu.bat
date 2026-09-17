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

REM ---- Resolve python command (python / py / python3) ----
set "PYEXE=python"
where %PYEXE% >nul 2>nul || set "PYEXE=py"
where %PYEXE% >nul 2>nul || set "PYEXE=python3"
where %PYEXE% >nul 2>nul || (
  echo [ERROR] python / py / python3 not found on PATH.
  echo         Install Python 3.12 or activate your venv/conda, then re-run.
  pause
  exit /b 1
)
echo Using python command: %PYEXE%

REM ---- Path (edit as needed) ----
set "SEQUOIA_DB=C:\Kronos\data\sequoia_v2.db"

REM ---- HuggingFace: download online on first run ----
set "HF_HOME=C:\Kronos\.hf_cache"
REM Use mirror if huggingface.co is slow or blocked; comment out next line if direct works
set "HF_ENDPOINT=https://hf-mirror.com"
REM Uncomment next line if you copied weights into HF_HOME and want fully offline
REM set "HF_HUB_OFFLINE=1"

REM ---- Data / text encoding ----
REM Default FinBERT is now Chinese finance model yiyanghkust/finbert-tone-chinese
set "SEQUOIA_FUSION=1"
set "SEQUOIA=1"
set "SEQUOIA_CPU_DATASET_PATH=.\data\sequoia_fusion_temporal"
set "SEQUOIA_CPU_MAX_SYMBOLS=300"
REM hash = offline deterministic hash fallback; auto = prefer FinBERT, degrade on fail
set "TEXT_ENCODER_MODE=auto"
set "TEXT_FINBERT_MODEL=yiyanghkust/finbert-tone-chinese"

REM ---- Training hyperparams (GPU scaled; 300 symbols) ----
set "FUSION_TRAIN_ITER=5000"
set "FUSION_VAL_ITER=200"
set "FUSION_BATCH_SIZE=64"
set "FUSION_EPOCHS=10"
set "FUSION_NUM_WORKERS=4"
set "SEQUOIA_DETERMINISTIC=0"

REM ---- 过拟合缓解（D）----
REM FUSION_WEIGHT_DECAY: AdamW 权重衰减（默认 0.01）
set "FUSION_WEIGHT_DECAY=0.01"
REM FREEZE_BACKBONE=1: 冻结 transformer/embedding/time_emb，仅训文本通道+head+dep_layer+norm
set "FREEZE_BACKBONE=1"

REM ---- Time splits (strict chronological) ----
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
%PYEXE% -u finetune\sequoia_db_loader.py || goto :error

echo == 2. Rebuild full-history text embeddings ==
%PYEXE% -u finetune\build_text_embeddings.py || goto :error

REM ---- Windows: single-GPU, skip torchrun, set dist env vars manually ----
set "USE_LIBUV=0"
set "MASTER_ADDR=localhost"
set "MASTER_PORT=29500"
set "RANK=0"
set "WORLD_SIZE=1"
set "LOCAL_RANK=0"

echo == 3. Fine-tune KronosFusion (GPU / single process) ==
%PYEXE% -u finetune\train_predictor_fusion.py || goto :error

echo == 4. Deterministic backtest (OOS, CPU) ==
set "MAX_WINDOW=250"
%PYEXE% -u finetune\backtest_predictor_fusion.py || goto :error

echo Task complete. OOS results in .\outputs\models\sequoia_predictor_fusion\backtest\
goto :eof

:error
echo [ERROR] Step failed, exit code %ERRORLEVEL%
exit /b %ERRORLEVEL%
