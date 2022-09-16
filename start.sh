#!/bin/bash
python -u main.py > run.out 2>&1 &
echo $!
