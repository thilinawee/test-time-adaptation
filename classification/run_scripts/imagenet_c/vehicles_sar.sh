#!/bin/bash


python test_time.py --cfg cfgs/imagenet_c/sar.yaml \
                         MODEL.ARCH resnet50_gn.a1h_in1k \
                         PRINT_EVERY 50 \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         PROJECT_NAME imagenet_c_test-time-adaptation \
                         RUN_NAME vehicles_sar_s5 \
                         FINAL_NUM_EX 150000 \
                         PARTIAL_CLASSES '[656, 675, 734, 555, 569, 717, 864, 867, 665, 407,
436, 468, 511, 609, 627, 661, 751, 817, 408, 573,
575, 803, 547, 820, 586, 802, 847, 561, 757, 829,
866, 603, 612, 690, 444, 671, 565, 705, 428, 791,
670, 870, 880, 833, 403, 510, 628, 724, 913, 814,
625, 472, 914, 554, 576, 484, 871, 780, 404, 895,
405, 417, 812, 450, 537, 466, 654, 779, 874, 830]'\