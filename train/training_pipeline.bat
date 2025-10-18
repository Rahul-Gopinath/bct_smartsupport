@echo off
setlocal

REM Set paths
set SEGMENTER=train/segmenter.py
set EMBEDDER=training/embedder.py
set CLUSTERER=training/clusterer.py
set INDEXER=training/indexer.py
set ANNOTATOR=training/annotator.py

set DIAGNOSER=inference/diagnoser.py

set LOG_DIR=logs\sample_logs
set SEGMENT_OUTPUT=training_segments.json
set EMBEDDING_OUTPUT=training_embeddings.npy

echo Running Drain3 segmentation...
python %SEGMENTER% @REM--log_dir %LOG_DIR% --output %SEGMENT_OUTPUT%

@REM echo Generating embeddings...
@REM python %EMBEDDER% @REM--input %SEGMENT_OUTPUT% --output %EMBEDDING_OUTPUT%

@REM echo Clustering embeddings...
@REM python %CLUSTERER% --output models/clustered_segments.json

@REM echo Indexing clustered segments...
@REM python %INDEXER%

@REM echo Annotating clusters with representative logs...
@REM python %ANNOTATOR%

@REM echo Training steps complete.

@REM python -m inference.diagnoser

endlocal