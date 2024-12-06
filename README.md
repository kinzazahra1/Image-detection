# YOLOv8 Fine-Tuning for Aquarium Dataset
## Project Overview
This project demonstrates how to fine-tune the YOLOv8 (You Only Look Once version 8) object detection model on a custom dataset, specifically the Aquarium Dataset from Kaggle. YOLOv8 is a state-of-the-art object detection algorithm, and this project aims to adapt its pretrained weights to detect objects in the aquarium dataset effectively.

The project involves training the YOLOv8 model, visualizing training metrics, evaluating model performance, and documenting the entire process. This repository includes all necessary files and instructions to replicate the results.

## Key Features
1. Pretrained YOLOv8 Fine-Tuning: Utilizes the pretrained YOLOv8 model (yolov8n.pt) to speed up training and improve performance.
2. Custom Dataset: Trained on the Aquarium dataset, containing images of aquatic creatures and objects.
3. Training Visualization: Includes graphs and metrics for training and validation loss, accuracy, precision, recall, and mAP.
4. Evaluation Metrics: Computes mean Average Precision (mAP) to evaluate the model's effectiveness.
5. Easy Reproducibility: Provides all necessary scripts and dependencies for seamless reproduction.

## Set up the dataset:

Download the Aquarium Dataset from Kaggle.
Place the dataset in the datasets/aquarium directory or update the paths in the data.yaml file.

## Dataset Structure
Ensure the dataset follows the YOLO format. Update data.yaml with the paths to the training and validation datasets. 

## Training
To fine-tune the YOLOv8 model on the Aquarium dataset:

1. Run the training script:

from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8 model
model.train(data='data.yaml', epochs=10, imgsz=640, batch=16, workers=4)

2. Training metrics such as loss, precision, recall, and mAP will be displayed after each epoch.

## Evaluation
Evaluate the model on the validation set with a confidence threshold of 0.5.

## Visualizations
The training metrics (loss, mAP, precision, recall) are saved in the runs directory. 

## Directory Structure
├── datasets/
│   ├── aquarium/  # Contains training and validation data
│       ├── train/
│       ├── val/
├── runs/  # Training runs and results
├── data.yaml  # Dataset configuration
├── requirements.txt  # Dependencies
├── train.py  # Training script
├── evaluate.py  # Evaluation script
└── README.md  # Project documentation 




