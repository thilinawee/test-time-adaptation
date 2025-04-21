
python shift_distance/calculate.py --cfg shift_distance/cfgs/tent.yaml \
                         PRINT_EVERY 50 \
                         MODEL.ARCH 'resnet50' \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         PROJECT_NAME accuracy_debug \
                         CKPT_SAVE_PATH /home/thilina/SSD2/thilina/test-time-adaptation-further_experiments/classification/shift_distance/ckpt/tent_freeze_s5/ \
                         CORRUPTION.TYPE '["gaussian_noise"]' \
                         PARTIAL_CLASSES '[213, 161, 173, 216, 165, 217, 177, 220, 168, 180]'