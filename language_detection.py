# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Implementation of a Language detection model using Naive Bayes Classifier

Training and Testing the Language detection model
"""

import pickle

from data import load_dataset
from utils import checkpoint_path
from naive_bayes import extract_features, train, predict


#reading the data text files in unicode and spliting into train and test sets
print("\t-------Loading Dataset-------")
X, y = load_dataset()  #generating the train set
print("Length of Dataset:", len(X))

#td-idf vectorizer and split data to the test and train sets
print("\t-------Extracting Features and Splitting Dataset-------")
train_x, test_x, train_y, test_y = extract_features(X,y)  #generating the train set
print("Length of Training set:", len(train_x))
print("Length of Test set:", len(test_x))

print("\t-------Start Training------")
classifier = train(train_x, train_y)
f = open(checkpoint_path, 'wb')
pickle.dump(classifier, f)
f.close()
print("Model saved:", checkpoint_path)    
print("\t-------End Training-------")

print("\t-------Start Testing------")
f = open(checkpoint_path, 'rb')
classifier = pickle.load(f)
f.close()

accuracy, confusion_matrix = predict(classifier,test_x,test_y)
print("Accuracy :", accuracy * 100)
print("\nConfusion Matrix:")
print(confusion_matrix)
print("\t-------End Testing------")                
