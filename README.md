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
We choose the linear regression model: y = wx + b. 
![y=wx+b](https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a})
### Step 4: Define The Loss Function





