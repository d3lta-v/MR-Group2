# Path Segmentation Training

## How to deploy

This code is designed to be deployed via Google Colab due to the good availability of GPU clusters and consistent build environment on Colab. 

1. Upload `track_yolo_train_notebook.ipynb` to Google Drive and open it in Colab. 
2. Spin up an T4 GPU runtime
3. Run `3_zip_datasets.py` to zip up the latest dataset and metadata into a single zip file that you can upload directly to Colab
4. Upload the zipped `dataset.zip` to Colab
5. Run the notebook
6. Once the training is complete, download `results.zip` from Colab and unzip it locally to obtain `best.pt`, which is the best performing weights
