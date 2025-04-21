import argparse
import os
import sys
import logging

import torch
import numpy as np
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import wandb
import pandas as pd

sys.path.insert(0, '/media/SSD2/thilina/test-time-adaptation-further_experiments/classification')
from models.model import get_model
from conf import cfg, load_cfg_from_args, get_num_classes, init_wandb, GlobalVar
from datasets.data_loading import get_source_loader, get_test_loader
from utils.indice_generator import generate_sample_indices, class_counts
from shift_distance.metrics import get_mean_embeddings, calc_mean_shift_distance, get_features \
    , get_cluster_centers, get_closest_cluster_centers

logger = logging.getLogger(__name__)

def get_tta_stats(run_id):
    
    # Initialize the W&B API
    api = wandb.Api()

    # Replace with your entity, project, and run ID
    # Access the run
    # run = api.run(f"{entity}/{project}/{run_id}")
    run = api.run(run_id)

    # Retrieve the run's history
    history = run.scan_history()

    # Convert the history to a DataFrame
    df = pd.DataFrame(history)
    df = df.dropna(subset=[f"avg_accuracy/{cfg.CORRUPTION.TYPE[0]}"])
    df = df.sort_values(by='custom_step', ascending=True)
    return df

def evaluate(description):
    load_cfg_from_args(description)
    # init_wandb(cfg)
    global_var = GlobalVar()

    df = get_tta_stats(cfg.RUN_ID)
    shift_distance_df = pd.DataFrame(columns=["custom_step", "shift_distance"])
    # exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model and its corresponding input pre-processing (if available)
    base_model, model_preprocess = get_model(cfg, num_classes, device)

    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess

    # @TODO implement the following
    sample_indices = generate_sample_indices(cfg.PARTIAL_CLASSES,
                                             cfg.CORRUPTION.NUM_EX_PER_CLASS,
                                             num_classes)

    _, source_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                      adaptation=cfg.MODEL.ADAPTATION,
                                      preprocess=base_model.model_preprocess,
                                      data_root_dir=cfg.DATA_DIR,
                                      batch_size=cfg.TEST.BATCH_SIZE,
                                      ckpt_path=cfg.MODEL.CKPT_PATH,
                                      use_clip=cfg.MODEL.USE_CLIP,
                                      num_samples=cfg.SOURCE.NUM_SAMPLES,
                                      percentage=cfg.SOURCE.PERCENTAGE,
                                      workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()),
                                      train_split=False,
                                      sample_indices=sample_indices)

    # logger.info("Source Class Counts", class_counts(source_loader))
    
    nodes, _ = get_graph_node_names(base_model)
    print(nodes)

    feature_layer = cfg.FEATURE_LAYER #"model.global_pool.pool"

    feature_list, label_list = get_features(base_model, 
                                            source_loader, 
                                            [feature_layer], 
                                            device=device,
                                            debug=False)
    
    
    clean_mean_embeddings = get_mean_embeddings(feature_list, label_list)
    del feature_list, label_list

    # -------------- Corrupted Data ----------------

    domain_sequence = cfg.CORRUPTION.TYPE

    for ii, domain_name in enumerate(domain_sequence):
        for severity in cfg.CORRUPTION.SEVERITY:

            corrupted_data_loader = get_test_loader(
                        setting=cfg.SETTING,
                        adaptation=cfg.MODEL.ADAPTATION,
                        dataset_name=cfg.CORRUPTION.DATASET,
                        preprocess=model_preprocess,
                        data_root_dir=cfg.TEST_DATA_DIR,
                        domain_name=domain_name,
                        domain_names_all=domain_sequence,
                        severity=severity,
                        num_examples=cfg.CORRUPTION.NUM_EX,
                        rng_seed=cfg.RNG_SEED,
                        use_clip=cfg.MODEL.USE_CLIP,
                        n_views=cfg.TEST.N_AUGMENTATIONS,
                        delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                        batch_size=cfg.TEST.BATCH_SIZE,
                        shuffle=False,
                        workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                        balanced=False,
                        oversampled_indices=sample_indices,
                        training=True # even though this is a test loader, subset selection is only implemented for training
            )

            ckpt_index = df["custom_step"].values.astype(int)
            for ckpt_ii in ckpt_index:
                
                if ckpt_ii == 0:
                    checkpoint = base_model
                else:
                    checkpoint = torch.load(f"{cfg.CKPT_SAVE_PATH}/ckpt_{ckpt_ii}.pt")
                    checkpoint.model_preprocess = model_preprocess

                feature_list, label_list = get_features(checkpoint, 
                                                        corrupted_data_loader, 
                                                        [feature_layer], 
                                                        device=device,
                                                        debug=False)
                
                cluster_centers = get_cluster_centers(feature_list, len(cfg.PARTIAL_CLASSES))
                
                shift_distance = calc_mean_shift_distance(list(clean_mean_embeddings.values()), cluster_centers)

                print(f"Checkpoint {ckpt_ii} Shift Distance: {shift_distance}")
                df.loc[(df["custom_step"] == ckpt_ii), "shift_distance"] = shift_distance
                df.loc[(df["custom_step"] == ckpt_ii), "avg_accuracy"] = df.loc[(df["custom_step"] == ckpt_ii), f"avg_accuracy/{domain_name}"]

    df.to_csv(f"{cfg.CKPT_SAVE_PATH}/shift_distance.csv")

if __name__ == '__main__':
    evaluate('"Evaluation.')
