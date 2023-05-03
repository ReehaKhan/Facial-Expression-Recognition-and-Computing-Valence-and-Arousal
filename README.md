# Facial-Expression-Recognition-and-Computing-Valence-and-Arousal
Implementation and Comparison of ResNet-18, MobileNet-V2 and EfficientNet-B1 on AffectNet dataset.

# Contents

# Abstract
Facial emotion recognition is the process of identifying a person’s emotional state by analyzing their facial expressions. Computing valence
and arousal in addition to facial expression recognition can provide a more comprehensive understanding of a person’s emotional state. In this
assignment, multi-task learning is implemented using three CNN architectures, which are MobileNet-V2, ResNet-18 and EfficientNet-B1. Transfer Learning is
used for EfficientNet-B1, while the other two were trained from scratch. The
dataset used is AffectNet that contains around 280K images annotated for expression, valence, arousal and lanmarks.
The results of the models are compared using many different evaluation metrics such as Accuracy, Cohens Kappa, Root Mean Square Error, Concordance Correlation Coefficent etc.. 
The best model found to be is ResNet-18, the results are given at the end.
