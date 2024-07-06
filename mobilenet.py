# %%
import math
import numpy as np 
import pandas as pd 
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
import seaborn as sns

from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.densenet import DenseNet201

import tensorflow as tf
from keras import Model

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from tqdm.notebook import tqdm_notebook as tqdm

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler

import random
import cv2

# %%
train_img_dir = '../input/airbus-ship-detection/train_v2/'
train_seg_csv = '../input/airbus-ship-detection/train_ship_segmentations_v2.csv'
test_img_dir = '../input/airbus-ship-detection/test_v2'
traincsv = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations_v2.csv')

# %%
traincsv.head()

# %%
c=[]
for i in (traincsv["EncodedPixels"].notnull()):

    if i==True:
        c.append(1)
    else:
        c.append(0)
        
traincsv["class"]=c

traincsv_unique = traincsv.drop_duplicates(subset=['ImageId'], keep='first')

print(traincsv_unique.head())
print("\n Shape of the Dataframe:",traincsv_unique.shape)

# %%
traincsv_unique = traincsv_unique.sort_values(by = ["class"])
traincsv_unique.reset_index(drop = True, inplace = True)

traincsv_unique = pd.concat([traincsv_unique.loc[:4999], traincsv_unique.loc[187556:]])

# %%
traincsv_unique["class"].value_counts()

# %%
IMAGE_SIZE = 128
paths = traincsv_unique["ImageId"]

# %%
batch_images = np.zeros((len(traincsv_unique["ImageId"]), IMAGE_SIZE, IMAGE_SIZE,3), dtype=np.float32)

for i, f in tqdm(enumerate(paths)):
  #print(f)
  img = Image.open(train_img_dir+f)
  img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
  img = img.convert('RGB')
  batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))

# %%
batch_images.shape

# %%
np.save("E:\\Resume",batch_images)

# %%
# batch_images1=batch_images.flatten()
# batch_images1=batch_images.swapaxes(1, 2).reshape(10000*128, 128*3)

# from numpy import savetxt
# savetxt('batch_images.csv', batch_images1, delimiter=',')

# %%
y = np.array(traincsv_unique["class"])
print(y)

# %%
np.save("class",y)

# %%
x_train_data , X_val, y_train_data , y_val = train_test_split(batch_images, y, test_size=0.2, random_state=42)

# %%
# model = DenseNet201(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),include_top=False, weights='imagenet', classes=2)


# for layers in model.layers:
#   layers.trainable = False

# x=model.layers[-1].output
# # x=tf.keras.layers.Dense(1024,activation='relu')(x)  
# #x=tf.keras.layers.Dense(512,activation='relu')(x) 
# x=tf.keras.layers.Flatten()(x)
# # x=tf.keras.layers.Dense(128,activation='tanh')(x)
# # x=tf.keras.layers.Dropout(0.4)(x)
# x=tf.keras.layers.Dense(64,activation='tanh')(x)
# x=tf.keras.layers.Dropout(0.4)(x)
# preds=tf.keras.layers.Dense(1,activation='sigmoid')(x) 


# model = Model(inputs = model.inputs, outputs = preds)

# %%
ALPHA = 1.0

def schedule(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

model = MobileNet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, alpha=ALPHA)

for layers in model.layers:
    layers.trainable = False

x = model.layers[-1].output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.6)(x)
preds = tf.keras.layers.Dense(1, activation='sigmoid')(x) 

model = Model(inputs=model.inputs, outputs=preds)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])

stop = EarlyStopping(monitor='val_binary_accuracy', patience=5, mode="max")
learning_rate = LearningRateScheduler(schedule)
reduce_lr = ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.2, patience=5, min_lr=1e-7, verbose=1, mode="max")

# Fit the model and capture the history
history = model.fit(x_train_data,
                    y_train_data,
                    batch_size=64,
                    epochs=20,
                    callbacks=[stop, reduce_lr, learning_rate],
                    validation_data=(X_val, y_val))

# Plot training & validation accuracy values
def plot_accuracy(history):
    print("Available keys in history.history:", history.history.keys())
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Plot training & validation loss values
def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Plot ROC curve
def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Print classification report
def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)

# Generating evaluation plots
def evaluate_model(model, X_val, y_val, history):
    predictions = np.round(np.squeeze(model.predict(X_val)))
    
    # Plot accuracy and loss
    plot_accuracy(history)
    plot_loss(history)
    
    # Plot ROC curve
    plot_roc_curve(y_val, predictions)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_val, predictions)
    
    # Print classification report
    print_classification_report(y_val, predictions)

# Call the evaluation function
evaluate_model(model, X_val, y_val, history)

# %%


# %%


# %%
def create_image_patches(image, patch_size=64):
    patches = []
    patch_positions = []
    img_height, img_width, _ = image.shape
    for y in range(0, img_height, patch_size):
        for x in range(0, img_width, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
                patch_positions.append((x, y))
    return patches, patch_positions

def preprocess_patches(patches, image_size=IMAGE_SIZE):
    preprocessed_patches = np.zeros((len(patches), image_size, image_size, 3), dtype=np.float32)
    for i, patch in enumerate(patches):
        img = Image.fromarray(patch)
        img = img.resize((image_size, image_size))
        img_array = preprocess_input(np.array(img, dtype=np.float32))
        preprocessed_patches[i] = img_array
    return preprocessed_patches

def draw_bounding_boxes(image, patches, patch_positions, predictions, patch_size=64, threshold=0.5):
    img_with_boxes = image.copy()
    for (x, y), prediction in zip(patch_positions, predictions):
        if prediction > threshold:
            img_with_boxes = cv2.rectangle(img_with_boxes, (x, y), (x+patch_size, y+patch_size), (0, 255, 0), 2)
    return img_with_boxes


# %%
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.applications.mobilenet import preprocess_input

# Define the test image directory
test_img_dir = '../input/airbus-ship-detection/test_v2/'

# Load test images
test_image_files = [os.path.join(test_img_dir, img) for img in os.listdir(test_img_dir)]

# Function to process and display the test images
def process_and_display_test_images(test_image_files, model, patch_size=64):
    for image_path in test_image_files:
        img = cv2.imread(image_path)
        patches, patch_positions = create_image_patches(img, patch_size=patch_size)
        preprocessed_patches = preprocess_patches(patches)
        predictions = model.predict(preprocessed_patches)
        img_with_boxes = draw_bounding_boxes(img, patches, patch_positions, predictions, patch_size=patch_size)
        
        # Display the image with bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title(f'Image: {os.path.basename(image_path)}')
        plt.axis('off')
        plt.show()

# Run the pipeline on the test images
process_and_display_test_images(test_image_files, model, patch_size=64)


# %%
predictions = np.round(np.squeeze(model.predict(X_val)))
predictions

# %%
i=random.randint(1,1500)

plt.imshow(X_val[i][:, :, 0],cmap='gray')
print("For {}th image:".format(i))
print("\tThe actual label class: ",y_val[i])
print("\tThe predicted label class: ",int(predictions[i]))

# %%
unscaled = cv2.imread("../input/airbus-ship-detection/test_v2/000f7d875.jpg")

image_height, image_width, _ = unscaled.shape
image = cv2.resize(unscaled,(IMAGE_SIZE,IMAGE_SIZE))
feat_scaled = preprocess_input(np.array(image, dtype=np.float32))
print("The predicted label",np.round(np.squeeze(model.predict(x = np.array([feat_scaled])))))
plt.imshow(unscaled)

# %%
