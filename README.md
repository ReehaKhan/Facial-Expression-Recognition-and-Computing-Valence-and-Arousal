# Facial-Expression-Recognition-and-Computing-Valence-and-Arousal
Implementation and Comparison of ResNet-18, MobileNet-V2 and EfficientNet-B1 on AffectNet dataset.

## Contents
1. [Abstract](1)
2. [Introduction](introduction)
3. [Set Up Instructions](setup)
4. [Data Set](dataset)
5. [Models](models)
6. [Results](results)

## Abstract {#1}
Facial emotion recognition is the process of identifying a person’s emotional state by analyzing their facial expressions. Computing valence
and arousal in addition to facial expression recognition can provide a more comprehensive understanding of a person’s emotional state. In this
assignment, multi-task learning is implemented using three CNN architectures, which are MobileNet-V2, ResNet-18 and EfficientNet-B1. Transfer Learning is
used for EfficientNet-B1, while the other two were trained from scratch. The
dataset used is AffectNet that contains around 280K images annotated for expression, valence, arousal and lanmarks.
The results of the models are compared using many different evaluation metrics such as Accuracy, Cohens Kappa, Root Mean Square Error, Concordance Correlation Coefficent etc.

## Introduction {#introduction}
Facial expression recognition is a research area in computer vision, with
numerous real-world applications such as human-computer interaction, emotion analysis, and security systems. Convolutional Neural Networks (CNNs)
have shown remarkable performance in facial expression recognition tasks in
recent years, due to their ability to learn complex features from images. CNNs
consist of multiple layers that learn features at different levels of abstraction
and are trained end-to-end using large datasets to improve their performance.
They are able to automatically learn features from raw images, without the
need for hand-crafted feature extraction, which can be time-consuming and
less effective. Furthermore, CNNs can learn discriminative features that are
specific to each expression, leading to more accurate recognition results.

Moreover, with the advancement of deep learning, multi-task learning has
been introduced. Multi-task learning is a technique in Convolutional Neural
Networks (CNNs) where a single model is trained to perform multiple tasks
simultaneously. In this approach, the model shares some of its parameters
across different tasks, allowing it to learn more robust and generalized features
that are relevant to all the tasks. This is in contrast to traditional CNNs, where
a separate model is trained for each task. The benefits of multi-task learning
include improved accuracy, reduced training time, and better model efficiency.

In this project, multi-task learning is implemented in Facial Emotion
Recognition while also predicting the Valence and Arousal values that range
between -1 and 1. Valence tells whether an expression is positive or negative
and arousal shows excitement/agitation or calm/soothing. This is done using
3 CNN Architectures, namely, MobileNet-V2, ResNet-18, and EfficientNet-B1.
The implementation and performance comparisons are discussed.

## Set Up Instructions {#setup}
To run these models, here are the instructions.

### Requirements
- python==3.10.11
- numpy==1.22.4
- pandas==1.5.3
- scikit-learn==1.2.2
- pytorch==2.0.0+cu118
- torchvision==0.15.1+cu118
- pillow==8.4.0
- matplotlib==3.5.1
- seaborn==0.12.2
- tqdm==4.65.0

### DataSet
The dataset can be downloaded from [here](http://mohammadmahoor.com/affectnet/). Please load the dataset on to your Google Drive from where it will be mounted.

### Running the Train Scripts
To train the model on your system, please follow the following steps:
1. Download the train script from the 'Train' folder and upload to Google Colab.
2. Mount the Google Drive and change the paths in the training notebook.
3. Run the code sequentially.
4. When training the model, one epoch takes one hour on standard Google Colab GPU T4. The training progress and the checkpoint at each epoch are saved as '.pth' file in the drive (keep in mind to change the paths).
5. After training is complete, the best validation performance model will be saved in your Google Drive.

### Running the Test Scripts
To test the model on the dataset, please follow the following steps:
1. From the 'Models' folder, open the models link.
2. Download the model you want to test.
3. Download the respective test script from the 'Test' folder.
4. Upload the test script and the model file to Google Colab.
5. Mount the Google Drive and change the paths in the testing notebook.
6. Run the code sequentially.

## DataSet {#dataset}
The dataset used in this project is [AffectNet](http://mohammadmahoor.com/affectnet/). AffectNet is a large facial
expression dataset with around 300K images manually labeled for the presence
of eight facial expressions along with the facial landmarks and the intensity
of valence and arousal. Each of these labels is stored in a separate annotation
file for each image. The dataset has 2 folders, one containing the train and
test data and another containing the validation data. Each dataset contains
the following:
- Eight emotion and non-emotion categorical labels (0: Neutral, 1: Happy, 2:
Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt). This is used
for the classification of expressions.
- Valence and Arousal values of the facial expressions in the continuous
domain, and are provided as floating point numbers in the interval [-1,+1].
These are used for the 2 regression tasks.
- Location of the Facial Landmarks. These are ignored in this work, as we use
Deep CNNs which are cable enough to learn features themselves.

In this project, the count of images in the train, test and validation set is as follows.
| Train | Test | Validation |
|-------|-------|-----------|
| 230120 | 57531 | 3999 |

## Models {#models}
The models chosen for this task are three, i.e., ResNet18, Mobilenet V2, and
Efficient Net B1. 
All three of these models are expected to perform 2 regression and 1 classification task. For this reason, a class is built that performs 3
FC layers in parallel. The last FC layer of ResNet18 and EfficientNet B1 and
the Classifier layer of MobileNet V2 is replaced with the CustomClassifier. The architecture of these networks are as follows. ResNet-18, MobileNet-V2 and EfficientNet-B1 in order.
<p align="center">
    <img src="https://github.com/ReehaKhan/Facial-Expression-Recognition-and-Computing-Valence-and-Arousal/blob/main/Architectures/resnet.jpg" width="200" height="500">
    <img src="https://github.com/ReehaKhan/Facial-Expression-Recognition-and-Computing-Valence-and-Arousal/blob/main/Architectures/mobnet.png" width="300" height="500">
    <img src="https://github.com/ReehaKhan/Facial-Expression-Recognition-and-Computing-Valence-and-Arousal/blob/main/Architectures/effnet.png" width="200" height="500">
</p>

The paramters and the transfer learning setting for the models are as follows.

|       | ResNet-18 | MobileNet-V2 | EfficientNet-B1 |
|-------|-----------|--------------|-----------------|
| Parameters | 11,181,642 | 2,236,682 | 6.525.994 |
| Transfer Learning | No | No | Yes |

### Loss Functions
The loss function used for classification is Cross Entropy Loss. Mean
Squared Error is used for regression.

### Training Parameters
|               | MobileNet V2 | Efficient Net B1 | ResNet18 |
|---------------|--------------|------------------|------------|
| Initial Learning Rate | 0.001 | 0.001 | 0.001 |
| Optimizer | Adam | Adam | Adam |
| Scheduler | StepLR | StepLR | StepLR |
| Scheduler Step Size | 4 | 4 | 4 |
| Scheduler gamma | 0.1 | 0.1 | 0.1 |
| Batch Size | 64 | 64 | 64 |
| Epochs | 10 | 8 | 10 |
| Time per Epoch (mins) | 57 | 65 | 50 |

### Training Graphs for ResNet-18
<p align="center">
    <img src="https://github.com/ReehaKhan/Facial-Expression-Recognition-and-Computing-Valence-and-Arousal/blob/main/Train/res%20acc%20graph.png" width="200" height="200">
    <img src="https://github.com/ReehaKhan/Facial-Expression-Recognition-and-Computing-Valence-and-Arousal/blob/main/Train/res%20loss%20graph.png" width="200" height="200">
    <img src="https://github.com/ReehaKhan/Facial-Expression-Recognition-and-Computing-Valence-and-Arousal/blob/main/Train/res%20val%20rmse%20graph.png" width="200" height="200">
    <img src="https://github.com/ReehaKhan/Facial-Expression-Recognition-and-Computing-Valence-and-Arousal/blob/main/Train/res%20aro%20rmse%20graph.png" width="200" height="200">
</p>

## Results {#results}
#### Quantitative Classification Results (Expression)

|                | **MobileNet V2** | **Efficient Net B1** | **ResNet18** |
| -------------- | ---------------- | --------------------- | ------------ |
| **Accuracy**         | 73.2             | 66.3                  | **76.7**          |
| **Cohens Kappa**     | 0.60             | 0.46                  | **0.66**          |
| **Krippendorffs Alpha** | -0.14            | -0.14                 | -0.14            |
| **AUC**              | 0.67             | 0.58                  | **0.72**          |
| **AUC-PR**           | 0.56             | 0.57                  | 0.56            |

The classification report:

|                  | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| **MobileNet V2**    |           |        |          |           |
| Neutral          | 0.58      | 0.86   | 0.69     | 14852     |
| Happy            | 0.89      | 0.89   | 0.89     | 27005     |
| Sad              | 0.72      | 0.31   | 0.43     | 5013      |
| Surprise         | 0.57      | 0.37   | 0.45     | 2799      |
| Fear             | 0.63      | 0.16   | 0.26     | 1274      |
| Disgust          | 0.46      | 0.10   | 0.16     | 766       |
| Anger            | 0.61      | 0.49   | 0.55     | 5043      |
| Contempt         | 0.00      | 0.00   | 0.00     | 799       |
| **EfficientNet B1** |           |        |          |           |
| Neutral          | 0.50      | 0.83   | 0.62     | 14982     |
| Happy            | 0.79      | 0.95   | 0.86     | 26889     |
| Sad              | 0.91      | 0.01   | 0.02     | 4921      |
| Surprise         | 1.00      | 0.00   | 0.00     | 2871      |
| Fear             | 0.00      | 0.00   | 0.00     | 1341      |
| Disgust          | 0.00      | 0.00   | 0.00     | 775       |
| Anger            | 0.75      | 0.0.   | 0.06     | 4974      |
| Contempt         | 0.00      | 0.00   | 0.00     | 778       |
| **ResNet 18**         |           |        |          |           |
| Neutral          | 0.65      | 0.81   | 0.72     | 14837     |
| Happy            | 0.89      | 0.92   | 0.91     | 26935     |
| Sad              | 0.69      | 0.50   | 0.58     | 5083      |
| Surprise         | 0.61      | 0.42   | 0.50     | 2817      |
| Fear             | 0.60      | 0.37   | 0.46     | 1317      |
| Disgust          | 0.56      | 0.22   | 0.32     | 762       |
| Anger            | 0.66      | 0.58   | 0.62     | 5018      |
| Contempt         | 0.25      | 0.00   | 0.01     | 762       |

#### Quantitative Regression Results (Valence)

|                              | MobileNet V2 | Efficient Net B1 | ResNet18 |
|------------------------------|--------------|------------------|----------|
| Root Mean Square Error       | 0.33         | 0.36             | **0.29** |
| Correlation                  | 0.80         | 0.75             | **0.83** |
| Sign Agreement Metric        | 0.80         | 0.76             | **0.83** |
| Concordance Correlation Co-efficient | 0.73 | 0.63             | **0.82** |

#### Quantitaitve Regression Results (Arousal)

|                             | MobileNet V2 | Efficient Net B1 | ResNet18 |
|-----------------------------|-------------|------------------|----------|
| Root Mean Square Error      | 0.26        | 0.29             | **0.24** |
| Correlation                 | 0.54        | 0.35             | **0.62** |
| Sign Agreement Metric       | 0.68        | 0.63             | **0.70** |
| Concordance Correlation Coefficient | 0.46 | 0.13             | **0.56** |

ResNet-18 has outperformed the other two models in all three tasks.

### Some classifications by ResNet-18
<p align="left">
    <img src="https://github.com/ReehaKhan/Facial-Expression-Recognition-and-Computing-Valence-and-Arousal/blob/main/Performance%20Evaluation/ResNet-18/res%20correctly%20classified%20images.png" width="500" height="300">
</p>
<p align="left">
    <img src="https://github.com/ReehaKhan/Facial-Expression-Recognition-and-Computing-Valence-and-Arousal/blob/main/Performance%20Evaluation/ResNet-18/res%20incorrectly%20classified%20images.png" width="500" height="300">
