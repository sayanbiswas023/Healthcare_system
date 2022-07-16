# HEALTHCARE SYSTEM ft. TEAM ICECREAM

### Team Members and Contributors- 
#### ANUBHAV RAJAK         @Anubhavrajak
#### ARUNIM BASAK          @Arunim10
#### SASWATI SHUBHALAXMI   @Sash002
#### VIKKY ANAND           @vikkyanand
#### SAYAN BISWAS          @sayanbiswas023

![alt text](https://i.pinimg.com/564x/c2/48/eb/c248eb292cb85b991512c02574723453.jpg)

<hr style="border:10px solid gray">

## THIS PROJECT INCLUDES: 
Improving Healthcare Access and Outcomes by integrating the applications of AI for
1. Primary care
2. Early detection of diseases
3. Providing patient support for managing diseases
4. Getting access to right healthcare
5. Emotional support

## FEATURES:
Multiple Diseases Prediction:
Prediction of the plausible disease of the person by taking inputs of various symptoms via a GUI.

Mental Health Awareness Chatbot:
Training a transformer-based attention model to generate relevant ‘replies’ to a previously defined ‘intent/emotion’ with every input statement from user.

Diagnosis of visually inspectable diseases: 
Creating an AI model to detect the occurrences of various diseases including cancers, cataracts, etc.

## DATASET DESCRIPTION:
Multiple Diseases Prediction:
Dataset : We will be using a dataset from Kaggle for this problem.

Mental Health Awareness Chatbot:   
Dataset : A chatbot framework requires structures where conversational intents are defined. A simple  and clean way to go about this is to use a JSON file: Intents	

Diagnosis of visually inspectable diseases: 
Dataset : Melanoma    |   Ocular Diseases

## MODEL DESCRIPTION
### Multiple Diseases Prediction:
We shall be using the dataset which consists of 41 classes of different diseases.
After cleaning the data for this model , we shall put in action various symptoms, which the users are suffering , as features for the model training. 
To train the model we will use Stacked Ensembling to get a joint venture prediction of the Decision Tree model, the Naive Bayes model, the RandomForest model and the Xgboost model. Using ensembling , we expect to get a very good accuracy of about 98 percent. 

### Mental Health Awareness Chatbot:
First, we will handle the initial NLP preprocessing.
Throughout that process, we shall be creating a list of sentences that could then be broken down further into a list of the stemmed words, with each sentence associated with an intent (a class).
After the natural language processing is handled, we look forward to proceed with building out the deep learning model, with Python.
The model will then be made user accessible by using the GUI Tkinter.

### Diagnosis of visually inspectable diseases:  
Data from the sources aforementioned are collected, cleaned and used for training. 
Training the model involves remarkable use of CNNs. 
The proposition is to use EfficientNet or MobileNet to enable decent efficiency and an incredibly fast model, suitable to use in a mobile application. 
The version 1 of the MVP uses Efficientnet B6 to detect malignancy of a skin lesion. The team is currently working on detection of stages of ocular diseases

### The GUI:
Efforts have been made to make the user interface understandable and easy to use, so as to reach the masses. 
Tkinter has been used to produce the entire interface. 
PROCESS OUTFLOW: 
The user will get to choose his/her symptoms from the list of options we provide on the application.
The four models(Decision Tree, Naive Bayes, RandomForest and XgBoost) having been ensemble would yield a result, thus predicting the the disease of the user.
The user would also get an option for getting mental health support on the same application; which is a chatbot named Me4U.  

## TECH STACK USAGE
1. Disease Predictor: Tensorflow, Scikit-Learn, Numpy, Pandas
2. Chatbot:Tensorflow, Numpy, Pandas, NLP, NLTK, Tkinter(For the GUI)
3. Early Disease Recogniser:Pytorch, Opencv-Python ,Matplotlib ,Pillow, Numpy
4. GUI:Tkinter

## INSPIRATION BEHIND THE PROJECT
1. The healthcare domain is not only one of the prominent research fields in the current scenario with the rapid improvement of technology and data but also a pressing issue for a country as demographically as large as India. While dealing with such a large amount of data is challenging, it is eased by Big Data Analytics. This work aims in automating the prediction, diagnosis and prognosis of medical conditions while making the technology available to the masses.
2. In a country with 1 doctor available per 750 people, being able to diagnose some fatal diseases even by our own, is likely to smoothen the medical procedures. 
Most of the patients die from negligence as they hesitate to visit a doctor even for the detection of a disease! Early and fast diagnosis of a plausible disease is important. This is where our AI Models and bot would come to rescue. We solely believe the ease of access and usage of this product, can prove to be of service. 
3. Mental health is the proverbial elephant in the room that no one wants to address. Corona is to blame for this massive spike in the number of cases. Chatbots are Natural Language Processing(NLP) based frameworks that interact with human . Built specifically to communicate with people struggling with mental problems.

<hr style="border:10px solid gray">
<hr style="border:10px solid gray">

## Instructions:
### 1. Clone
### 2. You'll need model weights...so....
![alt text](https://media.makeameme.org/created/gimme-money-or-6f5ba165c2.jpg)

Put the weights.h5 file in the 'skin_cancer' folder
### 3. Stay at root and run 
```	
	$ pip install -r requirements.txt
```
### 4. Run main.py


<hr style="border:10px solid gray">
<hr style="border:10px solid gray">

## Working Demo:

### 1. Home
![alt text](https://github.com/sayanbiswas023/Healthcare_system/blob/master/images/home.png?raw=true)

### 2. ChatBot
![alt text](https://github.com/sayanbiswas023/Healthcare_system/blob/master/images/chat.png?raw=true)

### 3. Disease Prediction
![alt text](https://github.com/sayanbiswas023/Healthcare_system/blob/master/images/disease.png?raw=true)

### 4. Skin Cancer Detection
![alt text](https://github.com/sayanbiswas023/Healthcare_system/blob/master/images/skin.png?raw=true)
