import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from pascal_voc_writer import Writer
from imageai.Detection.Custom import DetectionModelTrainer, CustomObjectDetection
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Mount Google Drive

# Constants
TRAIN_CSV_PATH = "C:/Users/krish/Downloads/train_ship_segmentations_v2.csv"
TRAIN_IMAGES_PATH = "C:/Users/krish/Downloads/train_v2"
VAL_IMAGES_PATH = 'validation/images/'
TRAIN_PATH = "train/"
VAL_PATH = "validation/"
TRAIN_IMG_DIR = TRAIN_PATH + "images/"
TRAIN_ANNOT_DIR = TRAIN_PATH + "annotations/"
VAL_IMG_DIR = VAL_PATH + "images/"
VAL_ANNOT_DIR = VAL_PATH + "annotations/"

# Read CSV
imageMasks = pd.read_csv(TRAIN_CSV_PATH)

# Decode Run Length Encoding (RLE)
def decodeRle(rleMask):
    rleMask = rleMask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (rleMask[0::2], rleMask[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(768 * 768, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(768, 768).T

# Generate mask image from masks list
def generateMaskImage(masksList):
    maskImage = np.zeros((768, 768))
    for mask in masksList:
        decodedMask = decodeRle(mask)
        maskImage += decodedMask
    return maskImage

# Get bounding box from mask image
def getBoundingBox(maskImage):
    labels = label(maskImage)
    coordinates = regionprops(labels)
    return coordinates

# Generate Pascal VOC annotations
def generateAnnotations(imageNames, imagesPath, annotationsPath):
    for i, imageName in enumerate(imageNames):
        print(f"Processing image {i+1}/{len(imageNames)}: {imageName}")
        image = cv2.imread(imagesPath + imageName)
        masksList = imageMasks.loc[imageMasks["ImageId"] == imageName]["EncodedPixels"]
        maskImage = generateMaskImage(masksList)
        coordinates = getBoundingBox(maskImage)
        pascalVoc = Writer(imagesPath + imageName, 768, 768)
        for coord in coordinates:
            pascalVoc.addObject('ship', coord.bbox[1], coord.bbox[0], coord.bbox[3], coord.bbox[2])
        pascalVoc.save(annotationsPath + imageName.split('.')[0] + ".xml")

# Create directory structure for YOLO training and validation data
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_ANNOT_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_ANNOT_DIR, exist_ok=True)

# Generate annotations for training and validation data
generateAnnotations(os.listdir(TRAIN_IMAGES_PATH), TRAIN_IMAGES_PATH, TRAIN_ANNOT_DIR)
generateAnnotations(os.listdir(VAL_IMAGES_PATH), VAL_IMAGES_PATH, VAL_ANNOT_DIR)

# Train YOLOv3 model
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setTrainConfig(object_names_array=["ship"], batch_size=4, num_experiments=50, 
                       train_from_pretrained_model="pretrained-yolov3.h5")  # Adjust this path if needed
trainer.trainModel()



# # Custom object detection
# detector = CustomObjectDetection()
# detector.setModelTypeAsYOLOv3()
# detector.setModelPath "models/detection_model-ex-002--loss-0012.548.h5")  # Adjust this path if needed
# detector.setJsonPath "json/detection_config.json")
# detector.loadModel()

# Detect objects and draw bounding boxes
def detectAndDrawBoundingBoxes(imagePath):
    detections = detector.detectObjectsFromImage(input_image=imagePath, output_image_path="detected.jpg", nms_threshold=0.2)
    image = cv2.imread(imagePath)
    for detection in detections:
        box_points = detection["box_points"]
        cv2.rectangle(image, (box_points[0], box_points[1]), (box_points[2], box_points[3]), (255, 0, 0), 2)
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# Test detection on test images
TEST_IMAGES_PATH = "C:/Users/krish/Downloads/test_v2"
test_images = os.listdir(TEST_IMAGES_PATH)

for test_image in test_images[:10]:  # Adjust the number of test images as needed
    detectAndDrawBoundingBoxes(TEST_IMAGES_PATH + test_image)
