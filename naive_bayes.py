# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Implementation of a Language detection model using Naive Bayes Classifier

Naive Bayes Classifier
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def extract_features(X, y, max_features=2000, test_size=0.2):
    """
        To apply tf-idf vectorizer and split data to the test and train data.
    
        X: corpus
        y: list of labels
        max_features: size of feature set
    """
    
    cv = TfidfVectorizer(max_features=max_features)
    X = cv.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle = True)
    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
        Train Naive Bayes Classifier.
        
        X_train: input data
        y_train: data labels
    """
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier


def predict(classifier, X_test, y_test):
    """
        Predict the accuracy of the model on the test set.
        
        classifier: trained naive bayes classifier
        X_test: test set datt
        y_test: test set labels
    """
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion= confusion_matrix(y_test, y_pred)
    
    return accuracy, confusion