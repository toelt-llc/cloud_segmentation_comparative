# Cloud segmentation comparative

# Comparative Study on Cloud Detection Algorithms

This repository contains a comprehensive study and comparison of various Cloud Detection Algorithms. The goal of this research is to provide insights into the efficiency, effectiveness, and reliability of these different algorithms when applied to various types of satellite imagery.

## Table of Contents

1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [Usage](#Usage)
4. [Algorithms Studied](#Algorithms-Studied)
5. [Datasets](#Datasets)
6. [Results](#Results)
7. [Contributing](#Contributing)
8. [License](#License)
9. [Contact](#Contact)

## Introduction

The field of deep learning continues to experience rapid advances, and numerous novel neural network architectures are being constructed within the domain of computer vision. These models excel in tasks such as classification, detection, tracking, and segmentation. However, the area where further development is still required is in the realm of Remote Sensing Images or Satellite Imagery. The main challenges stem from the fundamental differences between natural scene images and satellite imagery.

In satellite images, the represented information is predominantly overhead views, capturing only the tops or roofs of objects. On the other side, natural scene images provide multi-directional perspectives, resulting in a richer assortment of information. Edges in satellite images are especially informative in object representation, although they can potentially cause confusion for the models.

Moreover, satellite images are significantly influenced by factors such as shadows, seasonal variations, times of day and night that affect brightness and contrast, and even the altitude from which the image was captured. All of these elements can drastically alter the condition of the images. Therefore, interpreting satellite images presents a significant challenge for any computer vision model, and it is an area that calls for further research and improvement.

## Installation
The project require Python 3.7 or higher. The following dependencies are required to run the code in this repository:
Numpy
Pandas
Matplotlib
Scikit-learn
Scikit-image
OpenCV
Tensorflow
Torch

## Usage
The first step to run the code is to execut the scipts to download the datasets. The datasets are downloaded using the links in the Datasets section.
The scripts to download the datasets required for training can be found in source folder.

Once the datasets are downloaded, the next step is to run the notebooks to train the models. There is a notebook for each dataset. The notebooks can be found in the notebooks folder.

The next step is to run the notebooks to evaluate the models. By running Scoring.ipynb, the results of the model can be evaluated using task related metrics and with visual references. This notebook can be also found in the notebooks folder.

## Algorithms Studied
Here you can find a list of the papers that were studied in this research:

## Datasets
The datasets used in this research are the following:
1. [SPARCS](http://emapr.ceoas.oregonstate.edu/sparcs/)
2. [Landsat 8 Biome](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data)
3. [Sentinel 2 Cloud detection](https://mlhub.earth/data/ref_cloud_cover_detection_challenge_v1)
4. [Cloud 38 + 95](https://www.kaggle.com/datasets/sorour/95cloud-cloud-segmentation-on-satellite-images)
5. [CloudSEN12](https://www.scidb.cn/en/detail?dataSetId=f72d622ff4ea4fa18070456a98222b1a)

## Results

## Contributing

## License

## Contact
```bash

