# MBU-Net
Code for ICLR 2026 "Zeros can be Informative: Masked Binary U-Net for Image Segmentation on Tensor Cores". 
## Test Data Generation
For performance testing, `create_test_data.py` automatically generates dummy test data based on `layer_config.h`, where a layer is specified as binary or ternary. 
## Configuring for GPU Architecture
In `Makefile`, for example, `-arch=sm_80` is for A100, `-arch=sm_75` for RTX 2080Ti. 
## Compile and Run
`$ make unet && ./unet`. The profiling result should appear. 
