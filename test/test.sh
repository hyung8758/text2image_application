#!/bin/bash


for i in `seq $1`; do
    python test.py &
done
