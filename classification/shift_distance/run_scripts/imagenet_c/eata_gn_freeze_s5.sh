
python shift_distance/calculate.py --cfg shift_distance/cfgs/tent.yaml \
                         PRINT_EVERY 50 \
                         MODEL.ARCH resnet50_gn.a1h_in1k \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         CKPT_SAVE_PATH /home/thilina/SSD2/thilina/test-time-adaptation-further_experiments/classification/shift_distance/ckpt/eata_gn_freeze_ckpt_save/ \
                         CORRUPTION.TYPE '["gaussian_noise"]' \
                         PARTIAL_CLASSES '[213, 161, 173, 216, 165, 217, 177, 220, 168, 180]' \
                         RUN_ID rmit_is/shift_distance/1vzhn1ps \
                         FEATURE_LAYER model.global_pool.pool