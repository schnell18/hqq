#!/bin/bash

python bench.py 2>&1 | tee bench-$(date +%Y%m%d%H%M%S).log
