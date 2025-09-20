#!/bin/bash

algorithms=("tent")
partial_classes='[2, 9, 10, 11, 17, 21, 28, 37, 38, 39, 41, 42, 45]'
data_dir="<path_to_data_dir>"


for algo in "${algorithms[@]}"; do
    python test_time.py --cfg cfgs/domainnet126/experiments/${algo}_la_apx.yaml \
                                PRINT_EVERY 10 \
                                DATA_DIR $data_dir \
                                TRAIN_DATA_DIR $data_dir \
                                TEST_DATA_DIR $data_dir \
                                LOG_AVG_ACC True \
                                LOG_UNADAPTED_ACC True \
                                LOGIT_ADJUST.ALPHA 0.1 \
                                LOGIT_ADJUST.TAU 1.0 \
                                LOGIT_ADJUST.TYPE la \
                                LOGIT_ADJUST.WARMUP_STEPS 0 \
                                PARTIAL_CLASSES "${partial_classes}" 
done