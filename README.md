# Autoencoder-Image-Compression
Pytorch implementation for image compression and reconstruction via autoencoder

This is an autoencoder with cylic loss and coding parsing loss for image compression and reconstruction. Network backbone is simple 3-layer fully conv (encoder) and symmetrical for decoder. Finally it can achieve 21 mean PSNR on CLIC dataset (CVPR 2019 workshop).

![image](http://github.com/RobinWenqian/Autoencoder-Image-Compression/raw/master/Methodology.pdf)

You can download
training data from this url: https://drive.google.com/drive/folders/1wU1CO6WcQOraIaY2KSk7cRVaAXcm_A2R?usp=sharing

validation data: https://drive.google.com/drive/folders/113EcrAdcxfVqs8BVt4PZjwUEyVz7VVa-?usp=sharing

Organize your data with this structure:

Data/train/|image1.xxx|image2.xxx
        .

Data_valid/train/image1.xxx|image2.xxx
        .

You can train your own model via run_train.sh and modify config as your needs. Prediction for the valid data via run_test.sh
