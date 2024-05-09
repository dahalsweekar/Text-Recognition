# Nepali Text Recognition using small labeled dataset

## Introduction
This repository offers an implementation of text recognition where lstm network is used for seq-to-seq text recognition. Resnet-18 backbone is used for feature extraction.

## Features

  ### Training/Evaluation

| Flags  | Usage |
| ------------- | ------------- |
| ```--demo``` | run demo | 
| ```--train```  | start training	|                                                                   
| ```--eval```  | start evaluation |
| ```--visualize```  | visualize sample data  | 
| ```--plot```  | plot accuracy and loss | 
| ```--epoch```  | Set number of epochs  |
| ```--batch_size```  | set batch size  |
| ```--weights```  | set weights path |
| ```-dataset```  | set dataset directory  |
| ```--labels```  | set label path  |
| ```--test_image```  | set single image path  |

## Installation
  ### Requirements
    -Python3
    -Cuda

  ### Install
    1. git clone https://github.com/dahalsweekar/Text-Recognition.git
    
## Training 

 ```
 python services/train.py
 ```
## Models

  > Trained models are saved in ./models/

## Evaluation

 > A model must be trained and saved in ./models/ folder first
 ```
 python services/eval.py
 ```

