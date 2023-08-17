#!/bin/bash
# Download the dataset from Kaggle

# kaggle datasets -h
kaggle datasets download -d sorour/38cloud-cloud-segmentation-in-satellite-images -p /home/floddo/cloud_coverage_TOELT_SUPSI/Data/L8_95Cloud --unzip
kaggle datasets download -d sorour/95cloud-cloud-segmentation-on-satellite-images -p /home/floddo/cloud_coverage_TOELT_SUPSI/Data/L8_95Cloud --unzip
# ls /home/floddo/cloud_coverage_TOELT_SUPSI/Data/L8_95Cloud