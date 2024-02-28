<!-- # Cloud segmentation comparative -->
Official implementation of the paper _**BenchCloudVision**: A Benchmark Analysis of Deep Learning Approaches for Cloud Detection and Segmentation in Remote Sensing Imagery_
[ArXiv](https://arxiv.org/abs/2402.13918)

# Comparative Study on Cloud Detection Algorithms

This repository contains a comprehensive study and comparison of various Cloud Detection Algorithms. The goal of this research is to provide insights into the efficiency, effectiveness, and reliability of these different algorithms when applied to various types of satellite imagery.

[cloud_segmentation2.webm](https://github.com/toelt-llc/cloud_segmentation_comparative/assets/54261127/4a98e842-d674-4149-b7e8-68885dff4c3e)

## Table of Contents

1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [Algorithms Studied](#Algorithms-Studied)
4. [Datasets](#Datasets)
5. [Usage](#Usage)

<!-- 6. [Results](#Results)
7. [Contributing](#Contributing)
8. [License](#License)
9. [Contact](#Contact) -->

## Introduction

The field of deep learning continues to experience rapid advances, and numerous novel neural network architectures are being constructed within the domain of computer vision. These models excel in tasks such as classification, detection, tracking, and segmentation. However, the area where further development is still required is in the realm of Remote Sensing Images or Satellite Imagery. The main challenges stem from the fundamental differences between natural scene images and satellite imagery.

In satellite images, the represented information is predominantly overhead views, capturing only the tops or roofs of objects. On the other side, natural scene images provide multi-directional perspectives, resulting in a richer assortment of information. Edges in satellite images are especially informative in object representation, although they can potentially cause confusion for the models.

Moreover, satellite images are significantly influenced by factors such as shadows, seasonal variations, times of day and night that affect brightness and contrast, and even the altitude from which the image was captured. All of these elements can drastically alter the condition of the images. Therefore, interpreting satellite images presents a significant challenge for any computer vision model, and it is an area that calls for further research and improvement.

## Installation
The project require Python 3.7 or higher. The following dependencies are required to run the code in this repository:
- Numpy
- Pandas
- Matplotlib
- Scikit-learn
- Scikit-image
- OpenCV
- Tensorflow
- Torch

## Algorithms Studied
Here you can find a list of the papers that were studied in this research:
1. [A cloud detection algorithm for satellite imagery based on deep learning](https://www.sciencedirect.com/science/article/pii/S0034425719301294)
2. [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](http://arxiv.org/abs/1807.10165)
3. [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](http://arxiv.org/abs/1802.02611)
4. [Cloud-Net: An end-to-end Cloud Detection Algorithm for Landsat 8 Imagery](http://arxiv.org/abs/1901.10077)
5. [CloudX-net: A robust encoder-decoder architecture for cloud detection from satellite remote sensing images](https://www.sciencedirect.com/science/article/pii/S2352938520303803)
6. [Real-Time Flying Object Detection with YOLOv8](http://arxiv.org/abs/2305.09972)

## Datasets
The datasets used in this research are the following:
1. [SPARCS](http://emapr.ceoas.oregonstate.edu/sparcs/)
2. [Landsat 8 Biome](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data)
3. [Sentinel 2 Cloud detection](https://mlhub.earth/data/ref_cloud_cover_detection_challenge_v1)
4. [Cloud 38 + 95](https://www.kaggle.com/datasets/sorour/95cloud-cloud-segmentation-on-satellite-images)
5. [CloudSEN12](https://www.scidb.cn/en/detail?dataSetId=f72d622ff4ea4fa18070456a98222b1a)

For the training of the models only the first 2 datasets were used. 
The CloudSEN12 dataset was just explored but not used for hardware limitations.
The other datasets were used for the evaluation of the models.

## Usage
1. Clone the repository
```bash
git clone https://github.com/toelt-llc/cloud_segmentation_comparative.git
```
2. Install the dependencies
```bash
pip install -r requirements.txt
```
<!-- 2. Donwload the models weights
```bash
https://drive.google.com/drive/folders/1QgW_GihqJu6rxzJzhJWXUORWfUERSsaa?usp=sharing

``` -->
3. Download the datasets
The first step to run the code is to execute the scipts to download the datasets. The scripts can be found in the source folder and are named as follows:
```bash
python3 src/create_download_biome8.py
bash src/download_landsat8_biome.sh
bash src/download_kaggle_95Clouds.sh
python3 src/download_S2_mlhub.py
bash src/download_cloudSEN12.sh
```
The scripts were created for the larger datasets. The smaller datasets like the SPARCS can be easly downloaded manually from the links in the Datasets section.

4. Run the notebooks
Once the datasets are downloaded, the next step is to run the data exploration notebooks. The notebooks can be found in the notebooks folder. There is a notebook for each dataset. The function at the beginning will prepare the data for the training of the models. The data will be saved in the Data folder.

Once the data is prepared, the next step is to train the models.
There is a notebook for the training of the models on the SPARCS dataset and another one for the training of the models on the Landsat 8 Biome dataset. The notebooks can be found in the notebooks folder.

In each notebook there is a section where the model can be selected. The models are named as follows:
<!-- 1. UNet++
2. RS-Net
3. DeepLabV3+
4. CloudNet
5. CloudXNet -->
```bash
'unet'
'unet_plus_plus'
'rs_net'
'deep_lab_v3_plus'
'CXNet'
'cloud_net'
```
You can select the hyperparameters of the model in the same section, like learning rate, batch size, loss function, etc.

After that you just need to run the notebook and the model will be trained. The logs of the training will be saved in the logs folder. The models will be saved in the models folder by running the later cells in the notebook.


5. Run the notebooks for the evaluation of the models
The next step is to run the notebooks to evaluate the models. By running Scoring.ipynb, the results of the model can be evaluated using task related metrics and with visual references. This notebook can be also found in the notebooks folder.
There is a section where the model and the data setup on which the model was trained can be selected(for example if a model was trained on images with 3 channels or 4 channels, there is the need to specify that). After that the Scoring and Visual scoring sections should be run. Providing a comprehensive evaluation of the selected model.


6. Run the notebooks for YOLOv8
There is a separate notebook for YOLOv8, since it has a different training process. This notebook can be also found in the notebooks folder.
In the first section you can decide which dataset to use for the training. The options are:
```bash
'biome8'
'SPARCS'
```
After the configuration cell you can select some parameter fo the model, like number of epochs and batch size. The logs of the training will be saved in the runs folder.

When you have trained your YOLO model it will be automatically saved in the weights folder under runs. Then you can use the model for the evaluation by running the YOLOv8 Scoring notebook.

## Citation

If you use our repository or any of our implementation please cite us:
```
@misc{fabio2024benchcloudvision,
      title={BenchCloudVision: A Benchmark Analysis of Deep Learning Approaches for Cloud Detection and Segmentation in Remote Sensing Imagery}, 
      author={Loddo Fabio and Dario Piga and Michelucci Umberto and El Ghazouali Safouane},
      year={2024},
      eprint={2402.13918},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

This work belong to [TOELT](https://www.toelt-ailab.com/). It is open source and available for research and developement purposes only.

## Contact

- [fabio.loddo@student.supsi.ch](mailto:fabio.loddo@student.supsi.ch)
- [safouane.elghazouali@toelt.ai](mailto:safouane.elghazouali@toelt.ai)
