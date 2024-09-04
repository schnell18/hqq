#!/bin/bash

# export HF_HOME=/data/hugginface

if [ ! -d logs ]; then
    mkdir logs
fi
python bench.py 2>&1 | tee logs/bench-$(date +%Y%m%d%H%M%S).log
