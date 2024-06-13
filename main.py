from ultralytics import YOLO

from utils import split_save_train_val_test, masks_to_labels, save_masked_by_yolo, cropp_and_save_masked
from utils import copy_matching_files, read_water_meter


PREPARE_DATA_FOR_SEGMENT_TASK = False # Replace with True, if you want to prepare original data to YOLO segmentation model training
TRAIN_SEGMENTATION_MODEL = False # Replace with True, if you want to train YOLO segmentation model by yourself


PREPARE_DATA_FOR_DETECT_TASK = False # Replace with True, if you want to prepare data to YOLO detection model training
TRAIN_DETECTION_MODEL = False # Replace with True, if you want to train YOLO detection model by yourself


MAKE_PREDICTION_ON_IMAGE = False # Replace with True, if you want to make a prediction on your photo


if PREPARE_DATA_FOR_SEGMENT_TASK:

  path_to_dataset = 'data/OriginalData'
  path_to_save_train_val_test = 'data/YOLO_segmentation_data'

  # Copy the data into the directory structure required by YOLO
  split_save_train_val_test(path_to_dataset, path_to_save_train_val_test)

  # Convert all the masks to labels
  train_masks_path = 'data/YOLO_segmentation_data/labels/train_masks'
  train_labels_path = 'data/YOLO_segmentation_data/labels/train'
  masks_to_labels(train_masks_path, train_labels_path)

  val_masks_path = 'data/YOLO_segmentation_data/labels/val_masks'
  val_labels_path = 'data/YOLO_segmentation_data/labels/val'
  masks_to_labels(val_masks_path, val_labels_path)

  test_masks_path = 'data/YOLO_segmentation_data/labels/test_masks'
  test_labels_path = 'data/YOLO_segmentation_data/labels/test'
  masks_to_labels(test_masks_path, test_labels_path)


if TRAIN_SEGMENTATION_MODEL:
    
    # Load a nano-size model structure with trained weights
    model = YOLO('yolov8n-seg.pt')

    # Train on my data during 10 epochs
    results = model.train(data='yolo_segment.yaml', epochs=10)

# Visualising segmentation of your image
if False:
   segment_model = YOLO('models/segment_model.pt')
   image_path = 'data/OriginalData/images/id_4_value_352_676.jpg' # Change with path to meter photo you want to segment
   results = segment_model.predict(image_path, show = True)



if PREPARE_DATA_FOR_DETECT_TASK:
   
   # Apply masking to all the original images using trained segmentation model and save masked images
   path_to_images_dir = 'data/OriginalData/images'
   path_to_model = 'models/segment_model.pt'
   path_to_save_masked_images = 'data/masked_by_YOLO'
   save_masked_by_yolo(path_to_images_dir, path_to_model, path_to_save_masked_images)

   # Cropp numbers area from every masked image and save cropped images
   path_to_masked_dir = 'data/masked_by_YOLO'
   path_to_save_cropped = 'data/cropped_by_YOLO'
   cropp_and_save_masked(path_to_masked_dir, path_to_save_cropped)

   # Prepare data structure for training YOLO detection model
   path_to_cropped_images = 'data/cropped_by_YOLO'
   path_to_labels = 'data/annotated_numbers_for_YOLO_detection/labels'
   path_to_save_data = 'data/YOLO_detection_data'
   copy_matching_files(path_to_cropped_images, path_to_labels, path_to_save_data)


if TRAIN_DETECTION_MODEL:
    
    # Load the small-size model structure with pretrained weights
    model = YOLO("yolov8s.pt")

    # Train on my data during 20 epochs
    results = model.train(data="yolo_detect.yaml", epochs=20)


if MAKE_PREDICTION_ON_IMAGE:
   
   # Read meters readings on photo
   image_path = 'data/OriginalData/images/id_74_value_183_642.jpg' # Replace with a path to photo you want to read readings on
   segmentation_model_path = 'models/segment_model.pt'
   detection_model_path = 'models/detect_model.pt'
   path_to_save_predictions = 'predictions' # Replace with a path to folder where predictions will be saved 
   
   meters_readings = read_water_meter(image_path, segmentation_model_path, 
                                      detection_model_path, path_to_save_predictions)
    