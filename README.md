# ML_PM2.5_Prediction
## Introduction
This is my self-learning of the course Machine Learning 2019(Spring) in NTU. The instructor of this course is Hung-yi Lee. This is the homework 1. Check the website of the course: [Machine Learning 2019(Spring)](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)
## Our Main Goal 
The main purpose of this assignment is to practice how to implement the **Regression Model** to solve the problem. The main goal of this problem is to predict the value of the PM2.5 at 10th hour by the previous observation data in 9 hours.
### Step 1: Load .csv Data
In the train.csv, we have the real observation data from a weather station. What we have now is 18 features observed per hour in a day, and we only have 20 days of observation datas in each month. In total, we have 12x20x24 = 5760 sets of raw data. 
### Step 2: Sample Training Data From Raw Data
The raw data can be resampled every 10 hour, so that we will have 9 hours of training datas input and the real value of PM2.5 at 10th hour. After resampling, we should have 471x12 = 5652 sets of training input(x) and training output(y). 
### Step 3: Choose The Function Set (Model)
We choose the linear regression model: y = wx + b. In our case, this function should be expressed as **y** = **X** **w** + **b**
1. **y_head** = [y1 y2 y3 ... y5652]^T is the 5652x1 vector that represent the actually value of PM2.5 at 10th hour
2. **X** = [**x1** **x2** ... **x5652**]^T and **xi** = [feature1 at 1st hour, ... feature18 at 1st hour, feature1 at 2nd hour ... feature 18 at 9th hour] (i = 1~5652) is a 1x162 row vector. So **X** is a 5652x162 matrix
3. **w** = [w1 ... w162]^T is a 162x1 column vector
4. **b** = [b1 ... b5652] is a 5652x1 column vector
### Step 4: Define The Loss Function
We define the loss function as the square sum of all the error from a set of training data. That is, L(**w**) = sigma(**xi** **w** + bi - y_head_i)^2. 
### Step 5: Define The Gradient Of The Loss Function L(**w**)
1. Take the gradient of the loss function: gradient(L) = dL/d**w** = 2(sigma **xi**(**xi** **w** + bi - y_head_i))
2. Let yi = (**xi** **w** + bi) be the predicted value of Pm2.5 
3. Rewrite the equation in 1. We have gradient(L) = 2(sigma **xi**(yi - y_head_i)) = 






