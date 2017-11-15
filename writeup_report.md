
# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model_architecture.png "Model Architecture"
[image2]: ./examples/center_2017_11_12_22_34_48_435.jpg "Center Image"
[image3]: ./examples/center_2017_11_12_22_34_58_263.jpg "Recovery Image"
[image4]: ./examples/center_2017_11_12_22_34_58_743.jpg "Recovery Image"
[image5]: ./examples/center_2017_11_12_22_34_59_860.jpg "Recovery Image"
[image6]: ./examples/center_2017_11_12_22_34_57_651.jpg "Normal Image"
[image7]: ./examples/center_2017_11_12_22_34_57_651_flipped.jpg "Flipped Image"
[image8]: ./examples/left_2017_11_12_22_34_57_651.jpg "Left Camera"
[image9]: ./examples/right_2017_11_12_22_34_57_651.jpg "Right Camera"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* run6.mp4 which is a demo video of the car driving on the track
* this README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

In summary, the following steps can be found in the `model.py`:
* load csv data from disk, so obtain paths to image data
* build and compile model
* save the model into disk
* plot the model training and validation loss over epochs


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model (line 115 - 141) consists of the following:
* 5 convolutional layers
 * 5x5 filters and varying depths between 24 - 64 filters per layer
 * stride of 2
 * relu activation functions to introduce nonlinearity
 * droput of 25%
* 5 fully connected layers
 * 1 to 100 neurons per layer
 * relu activation except the output layer because the output needs both positive and negative values
* mean squared error for loss
* adam optimizer

The data is normalized into range between -1.0 to 1.0 in the model using a Keras lambda layer (code line 122).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. One dropout layer per convolutional layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 139).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I collected many different dataset, and tried use a combination of them, and the following combination seems to work the best in model learning:
* data1 - 1 lap of centerline driving with keyboard controls
* data2 - include 2.5 laps of driving that focus to stay in the middle of the lane
* data3 - one lap of driving in the opposite direction
* data4 - 2 more laps of really careful driving within lane
* data5 - a few curve driving
* data7 - some curves that the car struggle with and focused on recovery driving

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the model proposed in the [Nvdia paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I thought this model might be appropriate because it has multiple convolutional layer to be able to capture both low and high level features such as road edges, surface type and so on. In addition, it has one single vehicle control output, which is what the vehicle needs in the simulator to navigate.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to include dropout layers. Dropout layers make the validation error very similar to the training errors.

Then I attempted to increase the number of neurons in the fully connected layers in an attempt to lower error, but to no avail. It didn't seem to help at all.

The final step was to run the simulator to see how well the car was driving around track one.
* at first, the vehicle swirl a lot from left-to-right, and right-to-left. I realize that it may be due to I initially use keyboard to control the driving, which mostly have high steering angle value at very discreet steps. So I started collecting more data using mouse control.
* it still didn't quite help, I then realize that the steering angle that is set to 1 is too high. So I did trial-and-error, and drop it to 0.3, which seems to produce relatively more stable steering.
* Then, there were a few spots where the vehicle consistently fell off the track. I had to collect more driving data that focus on turning and curves.
* Finally, I had to try recovery driving and collect those data for the car to learn to veer back to the lane.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 115-141) consisted of 5 convolutional layers and 4 fully layers and 1 output layer. Here is a visualization of the architecture that is drawn straight from the [Nvdia paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center. Because when in these positions, I try to steer the vehicle back to the center in the training samples as well. These images show what a recovery looks like starting from the right road edge to the center of the road:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generate training samples of the car recovering from the opposite side of the road. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Another data augmentation technique is to use the images from both the right and left cameras. Because this would create even more data. For example, below are the two images from left and right side cameras. However, since the right(left) side of the images present quite different scene to the car, so the steering angle needs to be adjusted. I used a naive approach to adjust it by a constant factor comparing to the center steering angle. see code 59-73.

![alt text][image8]
![alt text][image9]

After the collection process, I had 9,762 number of data points. I then preprocessed this data by subtracting 255 from the image pixel values and minus 0.5 (i.e. (x / 255) - 0.5) so to bring the image pixel value range in between -1 and 1.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by a slight increase of the validation error at the last epoch.
