# MLSU & MSP-dataset
A Luminance-Aware Multi-Scale Network and Dataset for Polarization Image Fusion.\
Arxiv Link: [https://arxiv.org/abs/2508.16881](https://arxiv.org/abs/2510.24379v1)
# Network Architecture
![image](https://github.com/1hzf/MLS-UNet/blob/main/FIG/FIG/model.png)
# Dataset :
Please refer to the website "    " for the dataset related to the paper.
![image](https://github.com/1hzf/MLS-UNet/blob/main/FIG/FIG/dataset.png)
# Expirement
Python >= 3.8 \
Torch == 2.1.0 \
TorchVision == 0.16.0 \
TensorBoard == 2.13.0 \
scikit-image == 0.21.0 \
numpy >= 1.21 \
tqdm >= 4.60 \
torchinfo >= 1.8
# Table
Table1.Mean values of the metrics on the MSP dataset for the different fusion methods. (Red:optimal,blue: secondbest,green: third best).
![image](https://github.com/1hzf/MLS-UNet/blob/main/FIG/FIG/MSP-TT.png)
Table2.Mean values of the metrics on the PIF dataset for the different fusion methods. (Red:optimal,blue: secondbest,green: third best).
![image](https://github.com/1hzf/MLS-UNet/blob/main/FIG/FIG/PIF-TT.png)
Table3.Mean values of the metrics on the GAN dataset for the different fusion methods. (Red:optimal,blue: secondbest,green: third best).
![image](https://github.com/1hzf/MLS-UNet/blob/main/FIG/FIG/GAN-TT.png)
# Train
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py
```
# Gallery
![image](https://github.com/1hzf/MLS-UNet/blob/main/FIG/FIG/MSP1.png)
Figure 1:Example demonstration of the MSP dataset.
![image](https://github.com/1hzf/MLS-UNet/blob/main/FIG/FIG/PIF.png)
Figure 1:Example demonstration of the PIF dataset.
![image](https://github.com/1hzf/MLS-UNet/blob/main/FIG/FIG/GAN.png)
Figure 1:Example demonstration of the GAN dataset.
# Test
You need to put the dataset in the ./dataset/train/ .
You need to put the best.pt in the ./checkpoints/ .
```bash
python test.py
```
