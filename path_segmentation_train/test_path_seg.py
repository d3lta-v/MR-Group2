#!/usr/bin/env python3
"""
YOLOv11 Segmentation Inference Script
Runs batch prediction on images and exports results in multiple formats.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO


class YOLOSegmentationInference:
    """Class for running YOLOv11 segmentation inference and exporting results."""
    
    def __init__(self, weights_path, conf_threshold=0.25, iou_threshold=0.7, device=0):
        """
        Initialize the inference engine.
        
        Args:
            weights_path (str): Path to the trained model weights (best.pt)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            device (int/str): Device for inference (0 for GPU, 'cpu')
        """
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_names = self.model.names
        
        print(f"Loaded model from: {weights_path}")
        print(f"Class names: {self.class_names}")
    
    def predict(self, image_source, imgsz=640, save_visualizations=True, 
                output_dir="inference_results"):
        """
        Run inference on images and save all results.
        
        Args:
            image_source (str): Path to image file, directory, or glob pattern
            imgsz (int): Input image size
            save_visualizations (bool): Save annotated images
            output_dir (str): Output directory for results
            
        Returns:
            dict: Summary of inference results
        """
        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        viz_dir = output_path / "visualizations"
        masks_dir = output_path / "masks"
        crops_dir = output_path / "crops"
        json_dir = output_path / "json"
        txt_dir = output_path / "labels"
        
        if save_visualizations:
            viz_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        json_dir.mkdir(exist_ok=True)
        txt_dir.mkdir(exist_ok=True)
        
        # Run inference with streaming for memory efficiency
        print(f"\nRunning inference on: {image_source}")
        results = self.model.predict(
            source=image_source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=imgsz,
            device=self.device,
            stream=True,
            verbose=True,
            retina_masks=True  # High-resolution masks
        )
        
        # Process and save all results
        all_detections = []
        summary = {
            "total_images": 0,
            "total_detections": 0,
            "class_counts": {name: 0 for name in self.class_names.values()}
        }
        
        for idx, result in enumerate(results):
            img_name = Path(result.path).stem
            summary["total_images"] += 1
            
            # Process this image's results
            image_data = self.process_result(
                result, img_name, viz_dir, masks_dir, 
                crops_dir, json_dir, txt_dir, save_visualizations
            )
            
            all_detections.append(image_data)
            
            # Update summary statistics
            if result.masks is not None:
                num_objects = len(result.masks)
                summary["total_detections"] += num_objects
                
                for cls_id in result.boxes.cls.cpu().numpy():
                    class_name = self.class_names[int(cls_id)]
                    summary["class_counts"][class_name] += 1
            
            print(f"Processed {img_name}: {len(result.masks) if result.masks else 0} objects")
        
        # Save comprehensive results
        self.save_comprehensive_results(all_detections, summary, output_path)
        
        print(f"\n{'='*60}")
        print("Inference complete!")
        print(f"Total images processed: {summary['total_images']}")
        print(f"Total detections: {summary['total_detections']}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return summary
    
    def process_result(self, result, img_name, viz_dir, masks_dir, 
                      crops_dir, json_dir, txt_dir, save_viz):
        """Process a single image result and save in multiple formats."""
        
        image_data = {
            "image_name": img_name,
            "image_path": result.path,
            "image_shape": result.orig_shape,
            "detections": []
        }
        
        # Save visualization
        if save_viz:
            annotated = result.plot(
                conf=True,
                line_width=2,
                labels=True,
                boxes=True,
                masks=True
            )
            cv2.imwrite(str(viz_dir / f"{img_name}_annotated.jpg"), annotated)
        
        # Process each detection
        if result.masks is not None and len(result.masks) > 0:
            boxes = result.boxes
            masks = result.masks
            
            # Save YOLO format labels (normalized polygons)
            with open(txt_dir / f"{img_name}.txt", "w") as f:
                for i in range(len(masks)):
                    cls_id = int(boxes.cls[i].item())
                    conf = boxes.conf[i].item()
                    
                    # Get polygon points (normalized)
                    polygon_norm = masks.xyn[i]
                    
                    # Write to YOLO format: class_id x1 y1 x2 y2 ... xn yn
                    line = f"{cls_id}"
                    for point in polygon_norm:
                        line += f" {point[0]:.6f} {point[1]:.6f}"
                    f.write(line + "\n")
                    
                    # Store detection data
                    detection = {
                        "class_id": cls_id,
                        "class_name": self.class_names[cls_id],
                        "confidence": float(conf),
                        "bbox_xyxy": boxes.xyxy[i].cpu().numpy().tolist(),
                        "bbox_xywh": boxes.xywh[i].cpu().numpy().tolist(),
                        "polygon_normalized": polygon_norm.tolist(),
                        "polygon_pixels": masks.xy[i].tolist(),
                        "mask_area": float(masks.data[i].sum().item())
                    }
                    image_data["detections"].append(detection)
            
            # Save individual masks as binary images
            for i, mask in enumerate(masks.data):
                cls_name = self.class_names[int(boxes.cls[i].item())]
                mask_img = (mask.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(
                    str(masks_dir / f"{img_name}_mask_{i}_{cls_name}.png"),
                    mask_img
                )
            
            # Save combined mask (all objects)
            combined_mask = (masks.data.max(dim=0)[0].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(masks_dir / f"{img_name}_combined_mask.png"), combined_mask)
        
        # Save per-image JSON
        with open(json_dir / f"{img_name}.json", "w") as f:
            json.dump(image_data, f, indent=2)
        
        return image_data
    
    def save_comprehensive_results(self, all_detections, summary, output_path):
        """Save comprehensive results in multiple formats."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save complete JSON with all detections
        json_output = {
            "timestamp": timestamp,
            "model": str(self.model.ckpt_path),
            "confidence_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "class_names": self.class_names,
            "summary": summary,
            "detections": all_detections
        }
        
        with open(output_path / f"results_{timestamp}.json", "w") as f:
            json.dump(json_output, f, indent=2)
        
        # 2. Save CSV with all detections (flattened)
        csv_path = output_path / f"results_{timestamp}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_name", "detection_id", "class_id", "class_name",
                "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                "bbox_center_x", "bbox_center_y", "bbox_width", "bbox_height",
                "mask_area"
            ])
            
            for img_data in all_detections:
                for det_id, det in enumerate(img_data["detections"]):
                    writer.writerow([
                        img_data["image_name"],
                        det_id,
                        det["class_id"],
                        det["class_name"],
                        det["confidence"],
                        det["bbox_xyxy"][0],
                        det["bbox_xyxy"][1],
                        det["bbox_xyxy"][2],
                        det["bbox_xyxy"][3],
                        det["bbox_xywh"][0],
                        det["bbox_xywh"][1],
                        det["bbox_xywh"][2],
                        det["bbox_xywh"][3],
                        det["mask_area"]
                    ])
        
        # 3. Save summary report
        report_path = output_path / f"summary_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write("="*60 + "\n")
            f.write("YOLOv11 Segmentation Inference Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.model.ckpt_path}\n")
            f.write(f"Confidence Threshold: {self.conf_threshold}\n")
            f.write(f"IoU Threshold: {self.iou_threshold}\n\n")
            f.write(f"Total Images: {summary['total_images']}\n")
            f.write(f"Total Detections: {summary['total_detections']}\n\n")
            f.write("Detections by Class:\n")
            for class_name, count in summary['class_counts'].items():
                f.write(f"  {class_name}: {count}\n")
        
        print(f"Saved comprehensive results to {output_path}")


def main():
    """Main function to run inference."""
    
    # Configuration
    WEIGHTS_PATH = "best.pt"
    IMAGE_SOURCE = "../testimgs"  # Can be file, directory, or glob pattern
    OUTPUT_DIR = "test_results"
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.7
    IMGSZ = 320
    DEVICE = 'cpu'  # 0 for GPU, 'cpu' for CPU, 'mps' for Apple Silicon
    
    # Initialize inference engine
    inference = YOLOSegmentationInference(
        weights_path=WEIGHTS_PATH,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        device=DEVICE
    )
    
    # Run inference and save all results
    summary = inference.predict(
        image_source=IMAGE_SOURCE,
        imgsz=IMGSZ,
        save_visualizations=True,
        output_dir=OUTPUT_DIR
    )
    
    # Print summary
    print("\nClass Distribution:")
    for class_name, count in summary['class_counts'].items():
        if count > 0:
            print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
