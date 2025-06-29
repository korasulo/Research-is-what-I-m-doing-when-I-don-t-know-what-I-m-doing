![image](https://github.com/user-attachments/assets/7e282338-475b-460f-b8af-4c622e798d88)


# Efficient Image Classification with EfficientNetB0 on CIFAR-10
*A practical application of transfer learning with data augmentation for improved model generalization*

---

## Abstract
In this experiment, I implemented an image classification pipeline using EfficientNetB0 and the CIFAR-10 dataset. One of the biggest challenges I encountered was transforming CIFAR-10's tiny 32x32 images into the 224x224 input size required by EfficientNetB0 — this caused RAM-related issues and made the process infeasible when done in bulk. To overcome this, I designed a generator that resizes the images *on the fly* during training, which allowed me to run the model smoothly without exceeding memory limits. The pipeline includes data augmentation, transfer learning, and fine-tuning — all geared toward achieving strong performance on a resource-constrained setup.

---

## Introduction
The CIFAR-10 dataset, consisting of 60,000 color images in 10 categories, is a benchmark for image classification tasks. However, the small size of each image (32x32 pixels) poses a challenge when using deep architectures designed for higher-resolution inputs. Transfer learning, particularly using pre-trained models like EfficientNetB0, offers a solution by adapting robust feature extractors to new domains.  
Still, resizing all CIFAR-10 images to 224x224 ahead of time can be memory-intensive — this is a limitation I had to work around.

---

## Materials and Methods

### Dataset
- **Source**: `tensorflow.keras.datasets.cifar10`
- **Training set**: 50,000 images  
- **Test set**: 10,000 images  
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Preprocessing
- Labels were one-hot encoded using `to_categorical`.
- Instead of resizing all images beforehand (which previously caused out-of-memory errors on my machine), I implemented a **custom generator** that resized each image to 224x224 *during training*. This allowed me to train the model efficiently with limited hardware resources.

### Augmentation (Training only)
Implemented via `ImageDataGenerator`:
- Random rotation (±15°)
- Width and height shifts (10%)
- Zoom (up to 20%)
- Brightness adjustment
- Horizontal flipping
- All transformations were followed by EfficientNet-specific preprocessing.

### Model Architecture
- **Base**: `EfficientNetB0` (pre-trained on ImageNet, top layers removed)
- **Classifier head**:  
  - `GlobalAveragePooling2D`  
  - `Dense(256, activation='relu')`  
  - `Dropout(0.4)`  
  - `Dense(10, activation='softmax')`

### Training Phases
1. **Initial Training**: Base model frozen, trained for 30 epochs with Adam optimizer.
2. **Fine-Tuning**: Last 20 layers unfrozen, recompiled with reduced learning rate (`1e-5`), and trained for 30 more epochs.

### Implementation Stack
- TensorFlow 2.x  
- Keras (Sequential API)  
- NumPy

---

## Results
- The model showed consistent improvement in accuracy during the training and fine-tuning phases.
- Data augmentation significantly helped reduce overfitting, as observed through stable validation accuracy.
- By resizing images on the fly, I was able to complete training without running into RAM crashes, and performance was not compromised.
- After both training phases, the model was saved in `.h5` format for future use.


---

## Discussion
This experiment was not without obstacles. Initially, resizing all CIFAR-10 images from 32x32 to 224x224 in one go caused **memory overflow issues** on my local machine. The dataset, when fully resized and loaded into memory, became too large to handle. This bottleneck halted my progress until I came up with a solution: creating a **dynamic data generator** that resized each batch of images during training time.  
This allowed me to proceed without needing high-end hardware, and the model's performance was unaffected — a practical workaround to a common issue in deep learning with limited resources.  
It also illustrates how combining **transfer learning**, **data augmentation**, and **resource-aware engineering** can lead to successful outcomes even under constraints.

---


## Literature Cited
- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. arXiv preprint arXiv:1905.11946.  
- Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images*. University of Toronto.

