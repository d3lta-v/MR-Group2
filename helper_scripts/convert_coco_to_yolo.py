from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="dataset/coco/",
    save_dir="dataset/yolo/",
    use_segments=True, use_keypoints=False, cls91to80=False
)