#!/bin/bash

python test_time.py --cfg cfgs/imagenet_c/experiments/eata_gn_freeze.yaml \
                         PRINT_EVERY 50 \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet/DOGS_EXTENDED_S5 \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         PROJECT_NAME imagenet_c_test-time-adaptation \
                         RUN_NAME dogs_eata_gn_freeze_s5