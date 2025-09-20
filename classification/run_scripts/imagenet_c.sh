#!/bin/bash

algorithms=("tent")
severity=5
total_samples=150000
partial_classes='[203, 157, 241, 189, 162, 234, 178, 215, 196, 251, 167, 183, 209, 156, 228, 194, 171, 238, 185, 202, 219, 164, 247, 176, 153, 261, 198, 175, 266, 212, 159, 243, 187, 224, 170, 255, 191, 168, 206, 181, 233, 152, 217, 195, 174, 258, 163, 229, 184, 248, 161, 201, 177, 236, 190, 213, 166, 254, 179, 222, 158, 245, 192, 208, 173, 260, 186, 231, 165, 250, 200, 216, 182, 263, 155, 227, 169, 242, 188, 257, 204, 172, 239, 180, 264, 154, 220, 197, 267, 235, 151, 214, 265, 256, 193, 230, 207, 268, 262, 249]'
data_dir="<path_to_data_dir>"


for algo in "${algorithms[@]}"; do
    python test_time.py --cfg cfgs/imagenet_c/experiments/${algo}_la_apx.yaml \
                                PRINT_EVERY 50 \
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