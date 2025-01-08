#!/bin/bash

python logit_explosion/logits_main.py --cfg cfgs/imagenet_c/experiments/eata_gn_freeze.yaml \
                         PRINT_EVERY 50 \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         PROJECT_NAME logit_explosion \
                         RUN_NAME dogs_eata_gn_freeze_s5 \
                         CORRUPTION.TYPE "['gaussian_noise']" \
                         PARTIAL_CLASSES '[151, 152, 153, 154, 155, 
156, 157, 158, 159, 160, 
161, 162, 163, 164, 165, 
166, 167, 168, 169, 170, 
171, 172, 173, 174, 175, 
176, 177, 178, 179, 180, 
181, 182, 183, 184, 185, 
186, 187, 188, 189, 190, 
191, 192, 193, 194, 195, 
196, 197, 198, 199, 200, 
201, 202, 203, 204, 205, 
206, 207, 208, 209, 210, 
211, 212, 213, 214, 215, 
216, 217, 218, 219, 220, 
221, 222, 223, 224, 225, 
226, 227, 228, 229, 230, 
231, 232, 233, 234, 235, 
236, 237, 238, 239, 240, 
241, 242, 243, 244, 245, 
246, 247, 248, 249, 250, 
251, 252, 253, 254, 255, 
256, 257, 258, 259, 260, 
261, 262, 263, 264, 265, 
266, 267, 268]'