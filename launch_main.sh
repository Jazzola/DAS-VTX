#!/bin/bash

cd $HOME/DAS_VTX


timestamp=$(date +"%Y%m%d_%H%M%S")
python3 main.py \
  > "${timestamp}_stdout.log" \
  2> "${timestamp}_stderr.log"

