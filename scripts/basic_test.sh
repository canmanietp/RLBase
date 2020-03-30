#!/bin/bash

cd ..
cd src
python3 run.py --algorithms 'Q' 'MaxQ' --env='taxi' --num_trials=5