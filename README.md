# MLS-UNet & MSP-dataset
Polarization image fusion network based on multi-scale luminance-aware UNet & Multi-scene polarization dataset
# Expirement
Python >= 3.8 \
Torch == 2.1.0 \
TorchVision == 0.16.0 \
TensorBoard == 2.13.0 \
scikit-image == 0.21.0 \
numpy >= 1.21 \
tqdm >= 4.60 \
torchinfo >= 1.8

# Train
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

# Test
You need to put the dataset in the ./dataset/train/ .
You need to put the best.pt in the ./checkpoints/ .
```bash
python test.py
```
# Dataset :
Please refer to the website " https://pan.baidu.com/s/1ccNy96ImNmBPleKdzLeEJg?pwd=92sy " for the dataset related to the paper.
