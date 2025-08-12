#!/bin/bash

python test_time.py --cfg cfgs/domainnet126/tent.yaml \
                         PRINT_EVERY 50 \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets \
                         PROJECT_NAME domainnet126 \
                         FINAL_NUM_EX 150000 \
                         RUN_NAME tent \
                         LOG_AVG_ACC True \
                         LOG_UNADAPTED_ACC True \
                         PARTIAL_CLASSES '[2, 9, 10, 11, 17, 21, 28, 37, 38, 39, 41, 42, 45]'