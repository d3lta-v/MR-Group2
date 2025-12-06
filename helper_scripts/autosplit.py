# This script automatically splits a dataset into training, validation, and test sets, and ensures only annotated images are included.

from ultralytics.data.split import autosplit

autosplit(path="../dataset/yolo", weights=(0.8, 0.1, 0.1), annotated_only=True)
