# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./examples/nvidia_network.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py nvidiaModel.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
The model to be employed is essentially a regression neural network that takes an image as input and generate a steering angle as output. The model is trained in a supervised fashion with data provided by Udacity.

The Nvidia model was employed as my final model for its better(when compare to LeNet) empirical performance. This network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers.(model.py nvidiamodel function)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model
Training images are trimmed , flipped and augmented to increase the generalization ability of the model.

The model contains dropout layers in order to reduce overfitting and learn redundant representation. 

L2 regularization was also used to reduce overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 146). In data pre-processing, angle correction for side cameras was tuned to 0.25.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of all camera images with angle corrections so that the car can learn to recover if it's off the center of the road. Images and steering angles are flipped to mimic counter-clockwise training data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was test driven. That is after I had a working pipeline, I would continuously tune the model and get feedback in validation test.

My first step was to use the LeNet model which has proven its ability in former projects. Then I noticed that the car will lean to the right during test which is due to biased training data(clockwise data only). In order to solve this, I augmented the training data with images flipped and steering angles negated.

Then I found the car can stay on the center of the road for a while. But once it deviates from the center, the car can not recover and go offroad. In order to fix this problem, I need to introduce recovering training data. This is done by including side cameras into training data. Side cameras can be seen as center camera images whose orientations are off the center in a systematic way. So when appropriate corrections for their steering angle are given, the model can learn to recover from these data.

The appropriate corrections for side cameras come in two-folds. For the first part, I introduced fixed steering corrections for left and right cameras. The second part is regarding the images, I applied affine transformation to side camera images so that they look more like drawn from center camera.

But during training I noticed that the mse losses ocillate a lot, and in fact the model failed from time to time during simulation test. This is a sign of lack of network capacity in the LeNet model, so I altered the LeNet model to the Nvidia Model. 

The model trained smoothly and the final test in the simulator is good. The vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers.(model.py nvidiamodel function)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I mainly used the data set provided by Udacity which is proven to be enough to train a working model. Data augmentation methods are described in former answers. Same data augmentations are applied to test images in drive.py as well.

For the training process, I used a batch generator to resolve the memory issue introduced by this large data set. The batch generator will form a batch of training pairs and  randomly shuffle them. I used an adam optimizer so that manually training the learning rate wasn't necessary. The ideal number of epochs was 8 as evidenced by training loss will not decrease further after 8 epochs. 


