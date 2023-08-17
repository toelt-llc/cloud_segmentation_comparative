#!/bin/bash

# ls ../Data/CloudSEN12/ -la

mkdir ../Data/cloudSEN12/88__ROI_6464__ROI_6707
wget "https://download.scidb.cn/download?fileId=637e0d0586ce5f243f83c485&dataSetType=undefined&fileName=88__ROI_6464__ROI_6707.tar" -O 88__ROI_6464__ROI_6707.tar
tar -xf "88__ROI_6464__ROI_6707.tar" -C ../Data/cloudSEN12/88__ROI_6464__ROI_6707
rm 88__ROI_6464__ROI_6707.tar

mkdir ../Data/cloudSEN12/24__ROI_0549__ROI_0570
wget "https://download.scidb.cn/download?fileId=637daff286ce5f243f83c43b&dataSetType=undefined&fileName=24__ROI_0549__ROI_0570.tar" -O 24__ROI_0549__ROI_0570.tar
tar -xf "24__ROI_0549__ROI_0570.tar" -C ../Data/cloudSEN12/24__ROI_0549__ROI_0570
rm 24__ROI_0549__ROI_0570.tar


