#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/04273.jpg "Traffic Sign 1"
[image5]: ./test_images/04420.jpg "Traffic Sign 2"
[image6]: ./test_images/04491.jpg "Traffic Sign 3"
[image7]: ./test_images/04941.jpg "Traffic Sign 4"
[image8]: ./test_images/05057.jpg "Traffic Sign 5"
[image9]: ./chart1.png
[image10]:./Report/origin.jpg
[image11]:./Report/hisequ.jpg
[image12]:./Report/origin2.jpg
[image13]:./Report/transform1.jpg
[image14]:./Report/chart2.png
[image15]:./Report/chart3.png
[image16]:./Report/chart4.png

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bryan0806/Udacity-CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used python code to calculate summary statistics of the traffic
signs data set:

* The size of training set is  34799
* The size of test set is  12630
* The size of valid set is 4410
* The shape of a traffic sign image is  (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution.

You can see that the most seen sign is index 2 sign : Speed limit (50km/h)


![Training set data][image9]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

First, I found that in the data set, there are a lot of pictures are too dark or hard to see with our eyes. And second, the number of some signs has much more than others sign. For example, index 2, 1 and 38. 

Therefore, in the eighth code cell of the IPython notebook, I define 2 functions for future use:

1. Color normalization : Making R/G/B histogram equalization can help to make clear for some too dark or too bright images.

Here is an example of histogram equalization:

Origin:![origin][image10] 
After historgram equalization:![histogram equalization][image11]

2. Image Transformation: Some range of Rotate/Shear/Brightness Effect on image.

Here is an example of transform image:

Origin:![origin][image12]
After transform:![transform][image13]

In the 7th code cell of IPython notebook, I define Color normalization function.

This function make R/G/B value from 0-255 to 0-1, so it is easy and less error for future computation.

I let all image to apply this function before training.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data) 

To make my training data has same appearance chance of each sign, I look into the number of each unique sign then make it almost same.

Here is the final training set number of each sign. I also make the test set has almost same number of each sign.(It may be no necessary to do so.)

![equal times][image14]

![test_set][image15]

I actually try to split some data from training set to validation set. I saw some article about "cross validation". Using this way, I actually get a better accuracy from training. But I think validation should be some picture that never show up in training would be more challenging. Therefore, I still keep the validation set untouched.

My final training set had 86388 number of images. My validation set and test set had 4410 and 32208 number of images.

The 10th code cell of the IPython notebook contains the code for augmenting(image transform) the data set. I decided to generate additional data because more data could let the training more robust and it did rise my accuracy. To add more data to the the data set, I used opencv library to do specified range of rotate/affine/brightness effect on original data set.  

Here is an example of an original image and an augmented image:

Origin:![origin][image12]
After transform:![transform][image13]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 20th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout :0.5					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 4x4	    | 1x1 stride, valid padding, outputs 10x10x32     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Flatten     	| Input 5x5x32, outputs 800 	|
| Fully connected		| Input 800 outputs 120       	|
| RELU					|												|
| Fully connected		| Input 120 outputs 84       	|
| RELU					|												|
| Fully connected		| Input 84 outputs 43       	|
| softmax cross entropy		|         									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 26th cell of the ipython notebook. 

To train the model, I used an 0.01 learning rate. I did try other value, but did not get a better result.

Lower learning rate and rise the epoches may work but it took a long time to get the final result. Consider the efficiency, I decide to use still 0.01.

For the optimizer, I use AdamOptimizer. It has 2 parameter which I tuned it for a while. Beta1 =0.8 beta2=0.9. These 2 parameters are for learning rate decay. And I tried several values and decide to use 0.8 and 0.9 as my final value.



####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 25 cell of the Ipython notebook.

My final model results were:
* training set accuracy of 98.4%
* validation set accuracy of 94.4% 
* test set accuracy of 75.1%
    
    
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

First I found that using dropout let the learning become slower, but with more Epoch the accuracy become higher than before.
With no dropout, the validation accuracy is always lower than training accuracy. Therefore, I tune for several dropout. Finally I decide use total 1 dropout and 20 epoches to reach my best result.

If a well known architecture was chosen:
* What architecture was chosen? 
Lenet
* Why did you believe it would be relevant to the traffic sign application? 
We use Lenet on the number identification case before, and it works great. I think traffic sign has some number on it and think it should be work well on the same.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Traing accuarcy: 98.4%
Validation accuracy: 94.4%
Test accuracy: 75.1%

This prove my vision about this structure that it did work well on traffic sign classification.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image8] ![alt text][image5] ![alt text][image6] 
![alt text][image7] 

The second image might be difficult to classify because the upper part of image is bright and lower part is dark. The brightness difference make it difficult to classify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work     		| Road work 									| 
| Pedestrians			| Traffic signals     							|
| Wild animals crossing     			| Wild animals crossing 										|
| Priority road					| Priority road											|
| Yield	      		| Yield					 				|



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
This is closed to my testing accuracy which is about 75%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 33th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Road work (probability of 0.99), and the image does contain a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Road work  									| 
| .01     				| Traffic signals 										|
| .001					|Dangerous curve to the right										|
| .001	      			| Road narrows on the right				 				|
| .001				    | Beware of ice/snow      							|


For the second image, this is the only wrong answer for all my 5 prediction.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94         			| Traffic signals									| 
| .02     				|Pedestrians(correct answer)									|
| .01					|General caution										|
| .01	      			| Wild animals crossing				 				|
| .01				    | Road work     							|

For the rest of 3 images, this model all correctly predict the answer with 99%.
