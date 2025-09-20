#!/bin/bash

algorithms=("tent")
severity=5
total_samples=150000
partial_classes='[8, 58, 90, 13, 48, 81, 69, 41, 89, 85]'
data_dir="<path_to_data_dir>"


for algo in "${algorithms[@]}"; do
    python test_time.py --cfg cfgs/cifar100_c/experiments/${algo}_la_apx.yaml \
                                PRINT_EVERY 5 \
                                DATA_DIR $data_dir \
                                TRAIN_DATA_DIR $data_dir \
                                TEST_DATA_DIR $data_dir \
                                FINAL_NUM_EX $total_samples \
                                LOG_AVG_ACC True \
                                LOG_UNADAPTED_ACC True \
                                LOGIT_ADJUST.ALPHA 0.1 \
                                LOGIT_ADJUST.TAU 1.0 \
                                LOGIT_ADJUST.WARMUP_STEPS 0 \
                                CORRUPTION.SEVERITY [${severity}] \
                                LOGIT_ADJUST.TYPE la \
                                PARTIAL_CLASSES "${partial_classes}" 
done