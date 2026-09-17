@echo off
REM ============================================================
REM Robustness check: backtest the EXISTING trained fusion model
REM (sequoia_predictor_fusion / best_model) on the VALIDATION
REM period 2025-10-01~2026-03-31, which the model never saw
REM during gradient training (only used for epoch selection).
REM
REM Purpose: confirm the sign-flip finding (Run 2) holds on a
REM SECOND time window, not just 2026-04~09.
REM
REM Prereq: run run_sequoia_fusion_temporal_gpu.bat at least once
REM so .\data\sequoia_fusion_temporal\{train,val,test}_data.pkl
REM and text_embeddings.pkl exist. This script does NOT retrain
REM and does NOT rebuild embeddings; it only re-runs the backtest.
REM ============================================================

cd /d %~dp0

set "PYEXE=python"
where %PYEXE% >nul 2>nul || set "PYEXE=py"
where %PYEXE% >nul 2>nul || set "PYEXE=python3"
where %PYEXE% >nul 2>nul || (echo [ERROR] python / py / python3 not found on PATH. & pause & exit /b 1)
echo Using python command: %PYEXE%

set "SEQUOIA_DB=C:\Kronos\data\sequoia_v2.db"
set "HF_HOME=C:\Kronos\.hf_cache"
set "HF_ENDPOINT=https://hf-mirror.com"
set "SEQUOIA_FUSION=1"
set "SEQUOIA=1"
set "SEQUOIA_CPU_DATASET_PATH=.\data\sequoia_fusion_temporal"
set "SEQUOIA_CPU_MAX_SYMBOLS=300"
set "TEXT_ENCODER_MODE=auto"
set "TEXT_FINBERT_MODEL=yiyanghkust/finbert-tone-chinese"
set "FUSION_TRAIN_ITER=5000"
set "FUSION_VAL_ITER=200"
set "FUSION_BATCH_SIZE=64"
set "FUSION_EPOCHS=10"
set "FUSION_NUM_WORKERS=4"
set "SEQUOIA_DETERMINISTIC=0"
set "FUSION_WEIGHT_DECAY=0.01"
set "FREEZE_BACKBONE=1"
set "TRAIN_RANGE_START=2024-01-02"
set "TRAIN_RANGE_END=2025-09-30"
set "VAL_RANGE_START=2025-10-01"
set "VAL_RANGE_END=2026-03-31"
REM ---- Override test window to the VALIDATION period ----
set "TEST_RANGE_START=2025-10-01"
set "TEST_RANGE_END=2026-03-31"
set "MAX_WINDOW=250"
set "PYTHONPATH=%~dp0;%PYTHONPATH%"

echo == Backtest on VALIDATION period (model-unseen) to confirm sign flip ==
%PYEXE% -u finetune\backtest_predictor_fusion.py || (echo [ERROR] backtest failed & exit /b 1)
echo [done] Val-period backtest complete.
