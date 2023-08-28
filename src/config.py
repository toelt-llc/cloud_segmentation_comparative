# Configuration file for the project.
# Path: config.py
# Libraries:
from pathlib import Path
import os

# Global variables:
# ------------------------------------------------------------
project_root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
data_path = Path(project_root_path, "Data/")
source_path = Path(project_root_path, "src/")
notebooks_path = Path(project_root_path, "notebooks/")
models_path = Path(project_root_path, "models/")
results_path = Path(project_root_path, "results/")


# Models
saved_models_path = Path(models_path, "saved_models/")
train_history_path = Path(models_path, "train_history/")
checkpoint_path = Path(models_path, "checkpoints/")


# SPARCS:
sparcs_path = Path(data_path, "SPARCS/")
sparcs_raw_dir = Path(sparcs_path, "raw/")
sparcs_train_dir = Path(sparcs_path, "train/")
sparcs_valid_dir = Path(sparcs_path, "valid/")
sparcs_test_dir = Path(sparcs_path, "test/")

# YOLO:
yolo_data_dir = Path(data_path, "YOLO/")


# S2_mlhub:
s2_path = Path(data_path, "S2_mlhub/")
s2_im_path = Path(s2_path, "ref_cloud_cover_detection_challenge_v1_test_source/")
s2_labels_path = Path(s2_path, "ref_cloud_cover_detection_challenge_v1_test_labels/")
s2_train_dir = Path(s2_path, "train/")
s2_valid_dir = Path(s2_path, "valid/")
s2_test_dir = Path(s2_path, "test/")

# Biome8:
biome_path = Path(data_path, "L8_Biome8/")
# biome_raw_dir = Path(biome_path, "BC/")
biome_raw_dir = Path(biome_path, "raw/")

biome_train_dir = Path(biome_path, "train/")
biome_valid_dir = Path(biome_path, "valid/")
biome_test_dir = Path(biome_path, "test/")

# 95Clouds:
Cloud95_path = Path(data_path, "L8_95Cloud/")
Cloud95_38_train = Path(Cloud95_path, "38-Cloud_training/")
Cloud95_38_test = Path(Cloud95_path, "38-Cloud_test/")
Cloud95_additional = Path(Cloud95_path, "95-cloud_training_only_additional_to38-cloud/")


# cloudSEN12:
cloudSEN12_path = Path(data_path, "cloudSEN12/")

# Results:
# sparcs_results_path = Path(results_path, "SPARCS/")
# biome_results_path = Path(results_path, "Biome8/")
# s2_results_path = Path(results_path, "S2/")
# c95_results_path = Path(results_path, "95Cloud/")


# Check if the directories exist, and create them if they don't

# Models:
for directory in [models_path, saved_models_path, checkpoint_path, train_history_path]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# Data:
for directory in [sparcs_raw_dir, s2_path, biome_raw_dir, Cloud95_path, cloudSEN12_path, yolo_data_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
