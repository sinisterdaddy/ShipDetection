# %%
"""
## Overview
"""

# %%
"""
There are two most obvious network architectures to approach this competition: U-net and SSD. Each of them has pros and cons. In particular, U-net provides a relatively simple way to solve the competition challenge using image segmentation. However, this competition requires prediction of an individual mask for each ship rather than one mask for entire image. Therefore, some creative postprocessing may be needed, especially to separate ships with overlapping masks, if it is even possible. Another drawback is that the data is labeled with using pixelized bounding boxes rather than real ship masks, therefore the score of U-net based models is lowered. Meanwhile, implementation of SSD requires usage of rotating bounding boxes (https://arxiv.org/pdf/1711.09405.pdf) that is not common and, therefore, would take additional efforts for development of the model and the corresponding loss function. In addition, bounding boxes are not provided in this competition and must be generated based on the pixel masks. Nevertheless, this approach is expected to provide higher score than U-net, especially since the data is labeled based on pixelized bounding boxes (I expect organizers used SSD with rotating bounding boxes to label train and test data).
Since the first approach is more straightforward, I'll begin with presenting a kernel about U-net. In this post I will describe how to use pretrained ResNet34 to build a high accuracy image segmentation model. In particular, after training only a decoder for 1 epoch (15 min) on 256x256 rescaled images, the dice coefficient reaches ~0.8 (IoU ~0.67) that significantly outperforms all publicly available models posted so far in this competition. After training the entire model for 6 additional epochs with learning rate annealing, the dice coefficient reaches ~0.86 (IoU ~0.75). Due to the kernel run time limit, the model is further trained only for two epochs on 384x384 (dice ~0.87) followed by one epoch on 768x768 images. In an independent run I trained a model on 384x384 images for 12 epochs that boosted dice to 0.89 followed by training on full resolution images that increased dice further to 0.905.
"""

# %%
from fastai.conv_learner import *
from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# %%
"""
### Data
"""

# %%
PATH = './'
TRAIN = '/kaggle/input/airbus-ship-detection/train_v2'
TEST = '/kaggle/input/airbus-ship-detection/test_v2'
SEGMENTATION = '/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv'
PRETRAINED = '../input/fine-tuning-resnet34-on-ship-detection/models/Resnet34_lable_256_1.h5'
exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images

# %%
nw = 2   #number of workers for data loader
arch = resnet34 #specify target architecture

# %%
train_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
for el in exclude_list:
    if(el in train_names): train_names.remove(el)
    if(el in test_names): test_names.remove(el)
#5% of data in the validation set is sufficient for model evaluation
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)
segmentation_df = pd.read_csv(os.path.join(PATH, SEGMENTATION)).set_index('ImageId')

# %%
"""
One of the challenges of this competition is strong data unbalance. Even if only images with ships are considered, the ratio of mask pixels to the total number of pixels is ~1:1000. If images with no ships are included, this ratio goes to ~1:10000, which is quite tough to handle. Therefore, I drop all images without ships, that makes the training set more balanced and also reduces the time per each epoch almost by 4 times. In an independent run, when the dice of my model reached 0.895, I ran it on images without ships and identified ~3600 false positive predictions out ~70k images. The incorrectly predicted images were incorporated to the training set as negative examples, and training was continued. The problem of false positive predictions can be further mitigated by stacking U-net model with a classification model predicting if ships are present in a particular image (https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection - ~98% accuracy). 
I also noticed that in some kernels the dataset is tried to be balanced by keeping approximately the same number of images with 0, 1, 2, etc. ships. However, this strategy would be effective for such task as ship counting rather than training U-net or SSD.  One possible way to balance the dataset is creative cropping the images that keeps approximately the same number of pixels corresponding to a ship or something else. However, I doubt that such approach will effective in this competition. Therefore, a special loss function must be used to mitigate the data unbalance.
"""

# %%
def cut_empty(names):
    return [name for name in names 
            if(type(segmentation_df.loc[name]['EncodedPixels']) != float)]

tr_n = cut_empty(tr_n)
val_n = cut_empty(val_n)

# %%
def get_mask(img_id, df):
    shape = (768,768)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return img.reshape(shape)
    if(type(masks) == str): masks = [masks]
    for mask in masks:
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
    return img.reshape(shape).T

# %%
class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')
        super().__init__(fnames, transform, path)
    
    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 768: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    def get_y(self, i):
        mask = np.zeros((768,768), dtype=np.uint8) if (self.path == TEST) \
            else get_mask(self.fnames[i], self.segmentation_df)
        img = Image.fromarray(mask).resize((self.sz, self.sz)).convert('RGB')
        return np.array(img).astype(np.float32)
    
    def get_c(self): return 0

# %%
"""
The carrently availible on kaggle version of fastai has a bug in RandomLighting data agmentation class. It would be nice if kaggle updated fastai version to the last one, where this and other bugs are fixed.
"""

# %%
class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x  #add this line to fix the bug
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x

# %%
def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(20, tfm_y=TfmType.CLASS),
                RandomDihedral(tfm_y=TfmType.CLASS),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, 
                aug_tfms=aug_tfms)
    tr_names = tr_n if (len(tr_n)%bs == 0) else tr_n[:-(len(tr_n)%bs)] #cut incomplete batch
    ds = ImageData.get_ds(pdFilesDataset, (tr_names,TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    md.is_multi = False
    return md

# %%
"""
### Model
"""

# %%
"""
The model used in this kernel is inspired by a Carvana example from FastAI course (http://course.fast.ai/index.html). It is composed of a ResNet34 based encoder and a simple upsampling decoder. Similar to the original U-net, skip connections are added between encoder and decoder to facilitate the information flow at different detalization levels. Meanwhile, using a pretrained ResNet34 model allows us to have a powerful encoder capable of handling elaborated feature, in comparison with the original U-net, without a risk of overfitting and necessity of training a big model from scratch. The total capacity of the model is ~21M parameters. Before using, the original ResNet34 model was further fine-tuned on ship/no-ship classification task (https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection).
"""

# %%
cut,lr_cut = model_meta[arch]

# %%
def get_base():                   #load ResNet34 model
    layers = cut_model(arch(True), cut)
    return nn.Sequential(*layers)

def load_pretrained(model, path): #load a model pretrained on ship/no-ship classification
    weights = torch.load(PRETRAINED, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights, strict=False)
            
    return model

# %%
class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
    
class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)
        
    def forward(self,x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:,0]
    
    def close(self):
        for sf in self.sfs: sf.remove()
            
class UnetModel():
    def __init__(self,model,name='Unet'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]

# %%
"""
### Loss function
"""

# %%
"""
Loss function is one of the most crucial parts of the completion. Due to strong data unbalance, simple loss functions, such as Binary Cross-Entropy loss, do not really work. Soft dice loss can be helpful since it boosts prediction of correct masks, but it leads to unstable training. Winners of image segmentation challenges typically combine BCE loss with dice (http://blog.kaggle.com/2017/12/22/carvana-image-masking-first-place-interview/). Similar loss function is used in publicly available models in this completion. I would agree that this combined loss function works perfectly for Carvana completion, where the number of pixels in the mask is about half of the total number of pixels. However, 1:1000 pixel unbalance deteriorates training with BCE. 
If one tries to recall what is the loss function that should be used for strongly unbalanced data set, it is focal loss (https://arxiv.org/pdf/1708.02002.pdf), which revolutionized one stage object localization method in 2017. This loss function demonstrates amazing results on datasets with unbalance level 1:10-1000. In addition to focal loss, I include -log(soft dice loss). Log is important in the convex of the current competition since it boosts the loss for the cases when objects are not detected correctly and dice is close to zero. It allows to avoid false negative predictions or completely incorrect masks for images with one ship (the major part of the training set). Also, since the loss for such objects is very high, the model more effectively incorporates the knowledge about such objects and handles them even in images with multiple ships. To bring two losses to similar scale, focal loss is multiplied by 10. The implementation of focal loss is borrowed from https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c .
"""

# %%
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

# %%
class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()

# %%
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()

# %%
def dice(pred, targs):
    pred = (pred>0).float()
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)

def IoU(pred, targs):
    pred = (pred>0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)

# %%
"""
### Training
"""

# %%
m_base = load_pretrained(get_base(),PRETRAINED)
m = to_gpu(Unet34(m_base))
models = UnetModel(m)

# %%
models.model

# %%
sz = 256 #image size
bs = 64  #batch size

md = get_data(sz,bs)

# %%
# learn = ConvLearner(md, models)
# learn.opt_fn=optim.Adam
# learn.crit = MixedLoss(10.0, 2.0)
# learn.metrics=[accuracy_thresh(0.5),dice,IoU]
# wd=1e-7
# lr = 1e-2

# %%
##Adveat's Section

def decodeRle(rleMask):
    rleMask = rleMask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (rleMask[0:][::2], rleMask[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(768*768, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(768,768).T


def generateMaskImage(masksList):
    maskImage = np.zeros(shape=(768,768))
    for mask in masksList:
        decodedMask = decodeRle(mask)
        maskImage+=decodedMask
    return maskImage
import tensorflow as tf
from PIL import Image
filenames = ["0005d01c8.jpg", "00140e597.jpg","00113a75c.jpg","001dd855d.jpg","00269a792.jpg","001f3caca.jpg","0041d7084.jpg","002943412.jpg","002abd5df.jpg","002e85393.jpg"]
iou_threshold = 0.8
img_path = "/kaggle/input/airbus-ship-detection/train_v2"
trainCsv = pd.read_csv("/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv", index_col=0).dropna()
trainCsv = trainCsv.groupby("ImageId")[['EncodedPixels']].agg(lambda rle_codes: ' '.join(rle_codes)).reset_index()

class CheckOutputAndAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, interval=1, patience=5):
        super().__init__()
        self.all_target_images = [Image.open(os.path.join(img_path,file)) for file in filenames]
        self.all_target_masks = [generateMaskImage(trainCsv.loc[trainCsv["ImageId"]==file]["EncodedPixels"]) for file in filenames]
        self.all_target_instances = [self.extract_instances(mask) for mask in self.all_target_masks]
        self.interval = interval
        self.patience = patience
        self.wait = 0
        self.best_average_precision = -np.Inf
        self.best_epoch = 0
        self.predictions = None
        self.best_weights = None
        self.i=0
        
    def extract_instances(self, mask):
        labeled_mask = np.zeros_like(mask, dtype=int)
        label = 1
        instances = []
        height, width = mask.shape

        for y in range(height):
            for x in range(width):
                if mask[y, x] == 1 and labeled_mask[y, x] == 0:
                    component = []
                    stack = [(y, x)]

                    while stack:
                        cy, cx = stack.pop()
                        if labeled_mask[cy, cx] == 0:
                            labeled_mask[cy, cx] = label
                            component.append((cy, cx))
                            for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] == 1 and labeled_mask[ny, nx] == 0:
                                    stack.append((ny, nx))

                    instance_mask = np.zeros_like(mask)
                    for (iy, ix) in component:
                        instance_mask[iy, ix] = 1
                    instances.append(instance_mask)
                    label += 1

        return instances
    
    def compute_iou(self, pred_mask, target_mask):
        intersection = np.logical_and(target_mask, pred_mask).sum()
        union = np.logical_or(target_mask, pred_mask).sum()
        iou = intersection / union if union != 0 else 0
        return iou
    
    def compute_metrics(self, pred_masks):
        all_metrics = []

        for iter in range(len(self.all_target_images)):
            pred_instances = [instance for instance in self.extract_instances(pred_masks[iter])]
            target_instances = self.all_target_instances[iter]

            all_ious = []
            all_precisions = []
            all_recalls = []
            all_f1_scores = []
            all_average_precisions = []

            for target_mask in target_instances:
                best_iou = 0
                argmax_best_iou = -1

                for pred_mask in pred_instances:
                    iou = self.compute_iou(pred_mask, target_mask)
                    if iou > best_iou:
                        best_iou = iou
                        argmax_best_iou = pred_mask

                tp = best_iou >= iou_threshold
                fp = best_iou < iou_threshold and best_iou > 0
                fn = target_mask.sum() > 0 and best_iou == 0


                if tp:
                    precision, recall, f1, _ = 1.0, 1.0, 1.0, None
                else:
                    precision, recall, f1, _ = 0.0, 0.0, 0.0, None

                target_mask_flat = target_mask.flatten()
                best_pred_mask_flat = argmax_best_iou.flatten() if best_iou > 0 else np.zeros_like(target_mask_flat)
                average_precision = average_precision_score(target_mask_flat, best_pred_mask_flat)

                all_ious.append(best_iou)
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1_scores.append(f1)
                all_average_precisions.append(average_precision)

            metrics = {
                'Precision': np.mean(all_precisions),
                'Recall': np.mean(all_recalls),
                'Dice Score': np.mean(all_f1_scores),
                'Average Precision': np.mean(all_average_precisions),
                'Jaccard': np.mean(all_ious)
            }
            
            all_metrics.append(metrics)

        average_metrics = {}
        num_metrics = len(all_metrics)
        for d in all_metrics:
            for key in d:
                if key in average_metrics:
                    average_metrics[key] += d[key]
                else:
                    average_metrics[key] = d[key]

        for key in average_metrics:
            average_metrics[key] /= num_metrics

        return average_metrics
    
    def show_masks(self, pred_mask, target_mask):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(pred_mask, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(target_mask, cmap='gray')
        plt.title('Target Mask')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    def on_batch_begin(self,log = None):
        pass
    def on_epoch_end(self, epoch, logs=None):
        self.i += 1
#         print(f'Type of self.interval: {type(self.interval)}')
#         print(f'Type of epoch: {type(epoch)}')

        if (self.i) % self.interval == 0:
            try:
                predictions = self.model.predict(self.all_target_images)
                metrics = self.compute_metrics(predictions)

                # Display metrics
                print(f"\nMetrics at epoch {epoch + 1}:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")

                # Show masks for a sample image
                for sample_index in range(predictions.shape[0]):  
                    pred_mask = predictions[sample_index]
                    target_mask = self.all_target_masks[sample_index]
                    self.show_masks(pred_mask, target_mask)

                # Early stopping based on average precision
                current_avg_precision = metrics['Average Precision']
                if current_avg_precision > self.best_average_precision:
                    self.best_average_precision = current_avg_precision
                    self.best_epoch = self.i
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1

                if self.wait >= self.patience:
                    print(f"\nEarly stopping implemented")
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)

            except Exception as e:
                print(f"Error in on_epoch_end: {e}")


    
    def on_train_end(self, logs=None):
        print(f"\nBest Average Precision of {self.best_average_precision:.4f} at epoch {self.best_epoch}.")

check_output_accuracy_callback = CheckOutputAndAccuracy()

# %%
learn.freeze_to(1)

# %%
"""
Training only the decoder part for 1 epoch (15 min) leads to ~0.8 dice that outperforms all publicly available models in this competition.
"""

# %%

# Use the callback in the fit method
#learn.fit(lr, 2, wds=wd, cycle_len=1, use_clr=(5, 8), callbacks=[check_output_accuracy_callback])
learn.fit(callbacks=[check_output_accuracy_callback])

# %%
learn.save('Unet34_256_0')

# %%
"""
Unfreeze the model and train it with differential learning rate. The lr of the head part is still 1e-3, while the middle layers of the model are trained with 1e-4 lr, and the base is trained with even smaller lr, 1e-5, since low level detectors do not vary much from one image data set to another.
"""

# %%
lrs = np.array([lr/100,lr/10,lr])
learn.unfreeze() #unfreeze the encoder
learn.bn_freeze(True)

# %%
learn.fit(lrs,2,wds=wd,cycle_len=1,use_clr=(20,8))

# %%
learn.fit(lrs/3,2,wds=wd,cycle_len=2,use_clr=(20,8))

# %%
"""
The training has been run with learning rate annealing. Periodic lr increase followed by slow decrease drives the system out of steep minima (when lr is high) towards broader ones (which are explored when lr decreases) that enhances the ability of the model to generalize and reduces overfitting.
"""

# %%
learn.sched.plot_lr()

# %%
"""
Saved model can be ued for further training or for making predictions.
"""

# %%
learn.save('Unet34_256_1')

# %%
"""
### Visualization
"""

# %%
def Show_images(x,yp,yt):
    columns = 3
    rows = min(bs,8)
    fig=plt.figure(figsize=(columns*4, rows*4))
    for i in range(rows):
        fig.add_subplot(rows, columns, 3*i+1)
        plt.axis('off')
        plt.imshow(x[i])
        fig.add_subplot(rows, columns, 3*i+2)
        plt.axis('off')
        plt.imshow(yp[i])
        fig.add_subplot(rows, columns, 3*i+3)
        plt.axis('off')
        plt.imshow(yt[i])
    plt.show()

# %%
learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))

# %%
Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)

# %%
"""
The results are not ideal, but almost all ships are captured correctly even if the model is making the prediction on a very low resolution (256x256) images.
"""

# %%
"""
### Training (384x384)
"""

# %%
"""
Fortunately, modern convolutional nets support input images of arbitrary resolution. To decrease the training time, one can start training the model on low resolution images first and continue training on higher resolution images for fewer epochs. In addition, a model pretrained on low resolution images first generalizes better since a pixel information is less available and high order features are tended to be used.
"""

# %%
sz = 384 #image size
bs = 32  #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)

# %%
"""
Due to the kernel run time limit, the model was further trained only for two epochs on 384x384 (dice ~0.87) followed by one epoch on 768x768 images. In an independent run I trained a model on 384x384 images for 12 epochs that boosted dice to 0.89 followed by training on full resolution images that increased dice further to 0.905.
"""

# %%
learn.fit(lrs/5,1,wds=wd,cycle_len=2,use_clr=(10,8))

# %%
learn.save('Unet34_384_1')

# %%
"""
### Visualization
"""

# %%
learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))

# %%
Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)

# %%
"""
### Training (768x768)
"""

# %%
sz = 768 #image size
bs = 6  #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)

# %%
learn.fit(lrs/10,1,wds=wd,cycle_len=1,use_clr=(10,8))

# %%
"""
Training for just one epoch is insufficient to achieve high dice. However, if the training is continued, the dice can reach 0.90+.
"""

# %%
learn.save('Unet34_768_1')

# %%
"""
And finally, I put a picture (original image, prediction, ground truth) obtained by the model with dice 0.895 trained further on full resolution images in an independent run. Apart from one tiny ship in the last image, everything is captured. When I zoomed it in, it is really not clear if it is a ship of just a small island: I see only several white pixels, and there are several small islands under the water. Another interesting thing is that the model is able to capture details that it was not trained for. In particular, in 4-th image the model captures antennas (upper right ship) and the shape of ships, even if training set is composed of pixelized bounding boxes.
"""

# %%
"""
![1](https://image.ibb.co/mrqdze/Ship_Detection.png)
"""