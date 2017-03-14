
## **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals/steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/nvidiamodel.png "NVIDIA_model"
[image2]: ./pictures/model.png "Model"
[image3]: ./pictures/resize.png "Resize Image"
[image4]: ./pictures/brightness.png "Brightness Image"
[image5]: ./pictures/flip.png "Flip Image"

## Rubric Points
---

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
 
My project includes the following files:
* model.py contains the script to create and train the model.
* drive.py is used for driving the car in autonomous mode.
* model.h5 contains a trained convolution neural network model. 
* model.json contains architecture of the model. 
* Exploring_DataSet.ipynb was used as a framework for creating the                              
  finalized model.py file. I first used this file to explore the data set, 
  augment the data set, create the training set, and design/test the model 
  architecture. 
* writeup_report.md summarizes the results.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of five convolutional layers with three fully connected layers. The first three convolutional layers have a 3x3 kernel and a 2x2 stride. The last two convolutional layers have a 2x2 kernel and a 1x1 stride. The filter depths in the convolutional layers are between 24 and 64. The three fully connected layers follow the convolutional layers. After the last fully connected layer there is a single output value. 

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using the batch normalization function. The images were also normalized between a range of -0.5 and 0.5 before being fed into the model during the preprocessing stage since most of the steering angles fall around a steering angle of zero. 

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers with a dropout probability of 0.5 in order to reduce overfitting. The first dropout layer is added after the fifth convolutional layer and the second dropout layer is added after the first fully connected layer. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with a learning rate of 0.0001 since it is efficient and does not require manual tuning.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I ended up using the training data provided by Udacity. The dataset Udacity provided includes images that were collected from the left, center, and right camera of the car. Each camera image has its corresponding steering angle, throttle, brake, and speed values. I ended up using the left, center, and right camera images along with their corresponding steering angles for my data set. The udacity dataset is biased towards zero steering angles and right turns. To combat this problem I used the left and right cameras to simulate recovery data of the car driving from the edge of the track back to the center. This was done by adding an offset of +0.2 for the left camera images and -0.2 for the right camera images. Since there are more left turns in track 1 the model will make the car learn how to make left turns better than right turns. To account for this, I flipped all the images in the dataset horizontally and also inverted the corresponding steering angles by multiplying them by negative one. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture in my case was to use trial and error similar to what I did in the Traffic Sign Classifier Project. 

My first step was to take a look at the NVIDIA architecture described in this paper: https://arxiv.org/pdf/1604.07316v1.pdf as suggested by my mentor. My model is heavily inspired by the model in this paper. The graph of the NVIDIA model architecture is shown below: 

![alt text][image1]

I also took a look at the comma.ai paper: https://arxiv.org/pdf/1608.01230v1.pdf but decided to use the NVIDIA architecture as my starting point. The NVIDIA model is appropriate to follow because it tries to solve the same problem found in this project but at a larger scale and by using heavy augmentation techniques. 

I started by exploring the udacity data set (csv file) and analyzing the image files that were captured in the simulator for training. My mentor suggested reducing the size of the training images so I could train fast using my laptop's CPU instead of GPU. Reducing the size of the training images also allowed me to not use a python generator since my laptop was not using up a lot of memory by storing the data all at once. 

Just like in the Lane Lines project I decided to crop out scenery and just have the image contain the majority of the road. This would allow the model to just focus on the road during training and not learn features from scenery that are unnecessary. After cropping, I resized the image to be 32x64x3.

I also decided to add random brightness to the images. This was done to account for track 2 which contains different brightness conditions due to shadows. Since track 1 only has one type of brightness adding random brightness to the images would make the training more robust and would allow the car to perform better under different brightness conditions in other unseen tracks.

Furthermore, I put the images in different color spaces. I tried RGB, YUV, HSV, and HLS color spaces. The NVIDIA model uses YUV color space, but I experienced bad results using this color space since the car after training swerved around a lot and would go off the track. I did not see a benefit from using the HSV and HLS color spaces. Therefore, I decided to remain with the RGB color space. 

To create a more balanced dataset and combat the problem of right turn bias I decided to flip horizontally all the images in the original dataset and inverse their corresponding steering angles.

To simulate recovery data I added an offset of +0.2 to the left camera images and -0.2 for the right camera images. I initially started with an offset of +/-0.25 as suggested in the forums but after many training cycles the car would swerve(wobble) a lot on the track and was unstable even though it would complete the whole track successfully. I decided to reduce the offset to see if it would reduce the problem and I got better results by doing this. 

I used the NVIDIA model as a starting template. I decided to reduce the kernel sizes of the original NVIDIA model to 3x3 and 2x2 instead of 5x5 and 3x3 since I was working with smaller image sizes of 32x64x3. The NVIDIA model uses an image size of 66x200x3. For the strides, I decided to use a stride 2of x2 where the 3x3 kernel size was used and a stride of 1x1 where the 2x2 kernel size was used. The reason I did this is because I read online that as the kernel size decreases it is good practice to reduce the stride as well. I also decided to put a dropout layer with a dropout probability of 0.5 after the first fully connected layer to reduce overfitting. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I started training my model using the YUV color space for the images but my results were not satisfying as the car was swerving a lot and going off the track completely when I ran the simulator. I decided to stick with using the RGB color after seeing that I was getting better results and the car was able to complete more of the track effectively. 

I started using a learning rate of 0.001 for the adam optimizer, batch size of 128, and an epoch of 10 during testing. I did various trials where I changed the learning rate, batch size, and epoch number. I tried various combinations where I changed one of these variables while keeping the others constant. The learning rates I tried were 0.001, 0.0001, and 0.00001. The batch sizes I tried were 64 and 128. The epochs I tried were 5, 10, 15, and 20. I also decided to try using the HSV and HLS color spaces on the images but ended up just using the RGB color space after seeing that I was getting promising results with it. After various trials over many hours, I learned that increasing the learning rate resulted in better results because the car was actually able to complete a lap although it still would swerve alot. Unfortunately, the car was not able to complete various laps since sometimes it would go off the track after completing some laps. 

I ended up using a learning rate of 0.0001, batch size of 128, and an epoch of 20 as my successful parameters that allowed the car to complete a lap. What I was still struggling with was that the car was going off the track and into the water after the first sharp right turn. I read on the forums that having a high epoch number was giving people bad results and it was leading to overfitting. Therefore, I decided to lower the epoch number to 15 and then to 10. When I lowered the epoch number to 10 the car was able to drive successfully around the track for many laps without going off the track during the first sharp right turn. Furthermore, I read in the forums that dropouts should be added where there is the highest number of parameters in the layers. I checked my model summary and found that the highest number of parameters were after the last convolutional layer and the first fully connected layer so I made sure I added dropouts in those places. After doing this change the car was still able to drive around the track for many laps successfully. 

The last thing I needed to do was fix was the swerving(wobbling) of the car. To do this, I decided to lower the steering angle offset from +/- 0.25. I tried different lower offsets and ended up using +/- 0.2 since using lower offsets resulted in the car not completing turns correctly and going off the track. I also added the batch normalization function to the beginning of my model in addition to the original normalization I was already using on the images during preprocessing to see if I got better results. The swerving of the car was reduced effectively by applying this method. The car still swerves at times but not as drastically as before. I tried testing out again the HSV and HLS color spaces but they did not seem to improve my results so I decided to stay with the RGB color space. Reading the forums I learned how to set up a checkpoint to save all the epoch weights and try them one by one. This was the last step I did to ensure what epoch number would give me the best result with the model and training parameters I had in place. The final parameters that I chose were a learning rate of 0.0001, batch size of 128, and an epoch of 10. 

At the end of this whole approach, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes: 

**First layer:** Convolutional layer with 24x3x3 filters and stride 2x2        
                 RELU activation

**Second layer:** Convolutional layer with 36x3x3 filters and stride 2x2                   
              RELU activation
    
**Third layer:** Convolutional layer with 48x3x3 filters and stride 2x2      
             RELU activation
    
**Fourth layer:** Convolutional layer with 64x2x2 filters and stride 1x1       
              RELU activation
   
**Fifth layer:** Convolutional layer with 64x2x2 filters and stride 1x1      
             RELU activation                                                       
             Dropout of 0.5
             
**Sixth layer:** Flatten layer.
    
    
**Seventh layer:** Fully connected layer ---> 100 units                                      
               RELU activation                                               
               Dropout of 0.5
    
**Eighth layer:** Fully connected layer ---> 50 units                                     
              RELU activation
    
**Ninth layer:** Fully connected layer ---> 10 units                         
             RELU activation.
    
**Tenth layer:** Output layer ---> 1 unit                                                                  

Here is a visualization of the architecture:
![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To create the training set, I first cropped the left, center, and right camera images to remove most of the scenery and resized all these images to a smaller size of 32x64x3. I also registered the corresponding steering angles for the left, center, and right camera images.

![alt text][image3]

To augment the data set, I then added random brightness to all of the left, center, and right camera images.

![alt text][image4]


I also flipped images horizontally and invertedthe angles by multiplying them by -1.

![alt text][image5]


Furthermore, I made sure to add an offset of +0.2 to the left camera images and -0.2 to the right camera images. 
After the collection process, I had 48216 number of data points. I then preprocessed this data by normalizing the images between a range of -0.5 and 0.5 using min-max scaling. The data was also normalized using the batch normalization function in the beginning of my model. 


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or underfitting. I used an adam optimizer with a learning rate of 0.0001 so that manually training the learning rate was not necessary. The ideal number of epochs was 10 after going through many trials and determining that the car drove best around the track for many laps under these conditions. 


```python

```
