# Facial-Expression-Recognition-and-Computing-Valence-and-Arousal
Implementation and Comparison of ResNet-18, MobileNet-V2 and EfficientNet-B1 on AffectNet dataset.

## Contents

## Abstract
Facial emotion recognition is the process of identifying a person’s emotional state by analyzing their facial expressions. Computing valence
and arousal in addition to facial expression recognition can provide a more comprehensive understanding of a person’s emotional state. In this
assignment, multi-task learning is implemented using three CNN architectures, which are MobileNet-V2, ResNet-18 and EfficientNet-B1. Transfer Learning is
used for EfficientNet-B1, while the other two were trained from scratch. The
dataset used is AffectNet that contains around 280K images annotated for expression, valence, arousal and lanmarks.
The results of the models are compared using many different evaluation metrics such as Accuracy, Cohens Kappa, Root Mean Square Error, Concordance Correlation Coefficent etc.. 
The best model found to be is ResNet-18, the results are given at the end.

## Introduction
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

In this assignment, multi-task learning is implemented in Facial Emotion
Recognition while also predicting the Valence and Arousal values that range
between -1 and 1. Valence tells whether an expression is positive or negative
and arousal shows excitement/agitation or calm/soothing. This is done using
3 CNN Architectures, namely, MobileNet-V2, ResNet-18, and EfficientNet-B1.
The implementation and comparisons are discussed.
