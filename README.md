# Multiple-retinal-disease-classification-on-fundus-image-

## Abstract

<p align="justify">Early fundus screening is an inexpensive and effective way to prevent blindness caused by ophthalmic diseases in ophthalmology. Manual diagnosis is time-consuming, error-prone, and complicated in clinical settings due to a lack of medical resources, and it may cause the condition to worsen. Automated systems for the diagnosis of eye diseases with the help of artificial intelligence have become a hot research area in the medical field. Currently, most systems are designed to specifically detect eye diseases while humans can have more than one type of retinal disease in one eye. Therefore, it is necessary to develop an automated diagnostic system that can diagnose multiple diseases simultaneously. .</p>
<p></p>
<p align="justify">The proposed study presents a convolution neural network-based system for the diagnosis of various retinal diseases by fundus imaging. The proposed model system consists of 3 main parts: the data preprocessing phase, which includes data normalization and enhancement, the second phase is the modeling phase, and the last stage is the prediction phase. Recommended CNNs include ResNet 34, ResNet 50, Efficient Net, Inception V1, Inception V3, VGG 16. In the final phase the system will give the probability of all 9 diseases in each image. I validated the model by dividing the data into 3 sets: training set, Validation set and testing set, and measured performance using 4 different metrics: accuracy, recall, precision, and area under the curve (AUC).</p>

## Keywords
ocular disease; deep learning; fundus image; neural network; computer-aided diagnosis. 

## Pathologies

![state of eye disease](images/state_of_eye_disease.png)

<p align="justify">In the world, the number of people with vision impairments is expected to be 285 million, with 39 million blind and 246 million having impaired vision [7]. According to the World Health Organization (WHO), approximately 2.2 billion people have a close-up or distance vision problem and half of these cases could have been avoided or healed [8]. Uncorrected refractive errors (88.4 million), cataract (94 million), glaucoma (7.7 million), corneal opacities (4.2 million), diabetic retinopathy (3.9 million), and trachoma (2 million) all cause moderate-to-severe distance vision impairment or blindness, as do uncorrected presbyopia (826 million) [9].</p>

<p align="justify">Uncorrected refractive errors, cataracts, age-related macular degeneration, glaucoma, diabetic retinopathy, corneal opacity, trachoma, hypertension, and other causes of visual impairment are the most common [10]. In addition, studies have demonstrated that age related macular degeneration is the leading cause of blindness, particularly in developed nations, as it accounts for 8.7% (3 million people) of all blindness globally.</p>

<p align="justify">By 2040, the number of cases is expected to reach 10 million [1,3]. Recent studies [4,5] also demonstrated that 4.8% of the 37 million cases of blindness globally are attributable to diabetic retinopathy (i.e., 1.8 million persons). In 2000, more than 171 million people worldwide had diabetes, per the WHO [6]. By 2030, this number is anticipated to reach 366 million. Almost half of diabetic patients are unaware of their condition. Approximately 2% of diabetics become blind, and 10% develop severe visual impairment after 15 years. In addition, after 20 years of having diabetes, over 75% of patients will have some form of diabetic retinopathy.</p>

## Deep learning architecture

![proposal_system](images/proposal_system.png)

## Training Details

<!--[training](images/trainingdetails.png)-->

## Model Comparison


## Confusion matrix

<!--[ConfusionMatrix](images/ConfusionMatrix.png)

<p align="justify">As we can see in these confusion matrices. Inception does a better job of sorting items on the diagonal of the array, indicating the correct classification. If we had a perfect matrix, we would have to see number 50 in each cell on the diagonal. Therefore we have classifications with 80% of successes and others like for example the hypertension named with a 5 where we have only been able to correctly classify 22%. We have more than 50% of correct classifications in each class except hypertension and other pathologies with 22% and 32% respectively. However, despite the increase in data (through data augmentation), there are still features that have not been learned by the model.</p>

<p align="justify">As for the VGG, we can see how the data is a bit more scattered but we also have different classifications on the diagonal. As for the minority hypertension class, we can also see that there was an issue here as it was unable to classify too many images in this category.</p>

## Classification Output

![classificationoutput](images/classificationoutput.png)

<p align="justify">Finally we can see the output that each model generates and where we can visually check the classification result towards its ground truth. With this work, all the code related to the training and validation of the data, as well as the inference check to validate the output of the models, are delivered in this repo.</p>

<p align="justify">We can see, then, that the two models have the same classification for the same image, but if we analyze in detail the response of each output we can see that it is quite different.</p>

## Conclusions

- This project studies two deep learning models for the multiple classification of diseases.
- There is added complexity due to the multi-label and the initial data imbalance.
- We have seen that after the fine-tuning of the experiments we are able to obtain 60% accuracy on the validation set.
- The scenario is set for future applications, where the model could support the ophthalmologist during the capture of the fundus, and thus to classify pathologies faster.-->

# Implementation Details

## Dataset

The Dataset is part of the Retinal fundus multi disease image dataset (RFMID). In order to use the data you can download it from there: <https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification>

## Works on Python 3.9

The full list of packages used can be seen requirements.txt file.

All the training images must be in JPEG format and with 224x224px.

## Usage

### 1. Git clone repo
```cmd
git clone https://github.com/Hatran1412/Multiple-retinal-disease-classification-on-fundus-image-.git 
cd Multiple-retinal-disease-classification-on-fundus-image
``` 

### 2. Create environment
```cmd
conda create -n myenv python=3.9
conda activate myenv 
pip install -r requirements.txt
```
### 3. Download dataset
```cmd 
//create folder to saving dataset
mkdir dataset
//dowload dataset from kaggle
kaggle datasets download -d andrewmvd/retinal-disease-classification
//unzip folder dataset and delete zip file after unzip
unzip dataset/retinal_disease_classification.zip -d dataset/rfmid && del dataset/retinal_disease_classification.zip
```
### 4. Preprocessing data
Run the following cmd to treat the training and validation images:
#### a. Annotation csv
I only classified 9 classes so I edited the csv file.
```cmd
python re_label.py
```
#### b. Create folder for saving data after processing.
```cmd
cd dataset
mkdir data_preprocessing
cd data_preprocessing
mkdir training_set
mkdir testing_set
mkdir valdation_set
cd training_set
mkdir training_cropped
mkdir training_resized
mkdir training_augmented
cd validation_set
mkdir validation_cropped
mkdir validation_resize
cd testing_set
mkdir testing_cropped
mkdir testing_resize
```
#### c. Cropping image:

```cmd
//remove black pixels in image
bash crop.sh
```
#### d. Resizing iamge:
```cmd
//resize the images to 224 pixels
bash resize.sh
```
You can change the size to your liking by replacing 224 with another number in "image_resizing.py" file.

#### e. Augmentad image (if you don't want to use this step you can skip it)

```cmd
python image_augmentation_runner.py
```

This will generate the **augmented.csv** file.

#### f. Image to tf.Data conversion and .npy storage

```cmd
python.convert_to_numpy.py
```

Note that any changes in the images will need a re-run of this script to rebuild the .npy files.

### 5. CNN Models

#### a. Inception-v3

```cmd
-- Basic Run of the model
python CNN_models/inceptionv3/inception_v3_training_basic.py
```

```cmd
-- Enhanced Run of the model using Data Augmentation
python inception_v3_training_augmented.py
```

```cmd
python inception_v3_testing.py
```

#### b. Run VGG16

```cmd
-- Basic Run of the model
python vgg16_training_basic.py
```

- Download the VGG16 ImageNet weights from here: [weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5)
- 
```cmd
-- Enhanced Run of the model using Data Augmentation
python vgg16_training_augmented.py
```

```cmd
python vgg_testing.py
```

#### c. VGG19

```cmd
-- Basic Run of the model
python vgg19_training_basic.py
```

- Download the VGG19 ImageNet weights from here: [weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5)

```cmd
-- Enhanced Run of the model using Data Augmentation
python vgg19_training_augmented.py
```

```cmd
python vgg19_testing.py
```

#### d. ResNet50

```cmd
-- Basic Run of the model
python resnet50_training_basic.py
```

```cmd
-- Enhanced Run of the model using Data Augmentation
python resnet50_training_augmented.py
```

```cmd
python resnet50_testing.py
```

#### e. InceptionResNetV2

```cmd
-- Basic Run of the model
python inception_ResNetV2_training_basic.py
```


```cmd
-- Enhanced Run of the model using Data Augmentation
python. inception_ResNetV2_training_augmented.py
```

```cmd
python inception_ResNetV2_testing.py
```

## References



