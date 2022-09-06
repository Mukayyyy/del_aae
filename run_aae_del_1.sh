#!/usr/bin/bash

declare -a datasets=("ZINC")
# shellcheck disable=SC2068
for dataset in ${datasets[@]}; do
    python manage.py del --dataset "${dataset}" --ranking fnds --batch_size 256 --use_gpu
    python manage.py del --dataset "${dataset}" --ranking sopr --batch_size 256 --use_gpu
done
