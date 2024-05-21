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
| ```--dataset```  | set dataset directory  |
| ```--labels```  | set label path  |
| ```--test_image```  | set single image path  |

## Installation
  ### Requirements
    -Python3
    -Cuda

  ### Install
    1. git clone https://github.com/dahalsweekar/Text-Recognition.git

## Dataset sample:

![1](https://user-images.githubusercontent.com/99968233/231127471-4da5014a-113d-476e-9ee6-4dce754c3e40.jpg)
![2](https://user-images.githubusercontent.com/99968233/231127479-1813038e-5920-4055-871b-32783ac1b09d.jpg)
![3](https://user-images.githubusercontent.com/99968233/231127482-63898fa6-accb-451b-b534-76037f595aeb.jpg)
![4](https://user-images.githubusercontent.com/99968233/231127487-8604df08-460c-4f31-8903-c7f2f3e82ca5.jpg)
![5](https://user-images.githubusercontent.com/99968233/231127488-7bd5c88c-185b-4169-88c3-44928dc757de.jpg)
![6](https://user-images.githubusercontent.com/99968233/231127494-a4e67585-427d-4b2f-9827-c5141c362161.jpg)
![7](https://user-images.githubusercontent.com/99968233/231127498-1e3db845-d4d0-4d4b-a461-39bbb3965025.jpg)
![8](https://user-images.githubusercontent.com/99968233/231127500-4c78c08a-cabd-4171-95d0-b97d29dd6ba1.jpg)
![9](https://user-images.githubusercontent.com/99968233/231127504-2f06b4b5-3fc1-4fb4-897a-cc48a83e928d.jpg)
![10](https://user-images.githubusercontent.com/99968233/231127506-8b2163d2-5118-47d1-b370-c56b25e959c0.jpg)
![49](https://user-images.githubusercontent.com/99968233/231127618-761cf93a-41d2-44b2-b8c0-7bcfa4c5b5d2.jpg)
![50](https://user-images.githubusercontent.com/99968233/231127627-ccb97ed2-f2d7-44b8-b310-4a498f00df5b.jpg)
![51](https://user-images.githubusercontent.com/99968233/231127631-0699b264-aea8-4f7c-a54c-2ee338e99212.jpg)
![52](https://user-images.githubusercontent.com/99968233/231127643-f1435b08-77d7-4443-87a2-0e5984452d92.jpg)
![53](https://user-images.githubusercontent.com/99968233/231127648-f4853680-252b-4315-9477-44df0579a827.jpg)
![54](https://user-images.githubusercontent.com/99968233/231127649-7124b009-7264-4ea1-baeb-4ec6e170f2bb.jpg)
![55](https://user-images.githubusercontent.com/99968233/231127653-83f038bf-6fa0-4ca4-9240-46a3c805bc8a.jpg)
![56](https://user-images.githubusercontent.com/99968233/231127659-1e5e650b-f682-4e72-851a-678cd38abfb2.jpg)
![57](https://user-images.githubusercontent.com/99968233/231127662-19334891-036b-4a18-becb-2c39ee319106.jpg)
![58](https://user-images.githubusercontent.com/99968233/231127667-dd3eb2e3-9123-48a1-b356-bf67a0c433fd.jpg)
![59](https://user-images.githubusercontent.com/99968233/231127672-ec4f7e6a-e943-4820-9387-3f964c347b8b.jpg)
![60](https://user-images.githubusercontent.com/99968233/231127677-51162cdf-273e-4ddb-bfdf-5f21aa3dc8da.jpg)
![61](https://user-images.githubusercontent.com/99968233/231127682-8f05ad4a-33e0-4c43-8789-89c321b40942.jpg)
![62](https://user-images.githubusercontent.com/99968233/231127690-a1b13005-b0f5-4332-bc06-fc551e3445eb.jpg)
![63](https://user-images.githubusercontent.com/99968233/231127695-ae7ca4b4-bdf0-418b-86c9-d26b9929b12b.jpg)
![64](https://user-images.githubusercontent.com/99968233/231127699-933fee1f-44f3-4b7b-a3d9-1b34f51266f7.jpg)
![65](https://user-images.githubusercontent.com/99968233/231127705-d8032adb-07c0-4ca6-8d75-721f8631a4fe.jpg)
![66](https://user-images.githubusercontent.com/99968233/231127709-4176432a-f8b0-44a4-b873-89c481222db8.jpg)
![67](https://user-images.githubusercontent.com/99968233/231127710-58ee070d-f1d0-41c9-9763-7865cce372f2.jpg)

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

