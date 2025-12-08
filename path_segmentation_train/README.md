# Path Segmentation Training

## How to deploy

This code is designed to be deployed via Google Colab due to the good availability of GPU clusters and consistent build environment on Colab. 

1. Upload `track_yolo_train_notebook.ipynb` to Google Drive and open it in Colab. 
2. Spin up an T4 GPU runtime
3. Run `3_zip_datasets.py` to zip up the latest dataset and metadata into a single zip file that you can upload directly to Colab
4. Upload the zipped `dataset.zip` to Colab
5. Run the notebook
6. Once the training is complete, download `results.zip` from Colab and unzip it locally to obtain `best.pt`, which is the best performing weights

## Convert weights from PyTorch format to TensorRT format

Follow [this tutorial](https://docs.ultralytics.com/guides/nvidia-jetson/#run-on-jetpack-512) based on your JetPack version. 

Ensure that you've run this command after following the tutorial to ensure that you're able to export TensorRT. This will take about 5 minutes to run on the Jetson NX Xavier.

```bash
sudo apt install tensorrt nvidia-tensorrt-dev python3-libnvinfer-dev
```
