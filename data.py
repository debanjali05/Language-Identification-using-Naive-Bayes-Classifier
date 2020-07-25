# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Implementation of a Language detection model using Naive Bayes Classifier

Reading Dataset
"""

import re
import os
import string
import codecs

from utils import lang, dataset_path, dataset_lang_path

def preprocessing(line):
    """
        data preprocessing step: 
            1) Convert to lowercase
            2) Remove line number, digits, punctuations and extra spaces
    
        line: each sentence from the dataset (string)
    """
    
    translate_table = dict((ord(char), None) for char in string.punctuation)
    
    line = line.lower() #converting the text to lowercase 
    line = re.sub(r"\d+", "", line) #removing any digits present in the text
    line = line.translate(translate_table)  #removing all punctuations
    line = re.sub(' +',' ',line) #removing extra spaces
     
    return line #preprocessed text

def build_dataset(path,language):
    """
        Building the dataset for individual language
    
        path: path to the text
        language: language label (string)
    """
    
    language_set = []
    
    #Reading the data text files in unicode
    with codecs.open(path,"r","utf-8") as filep:
             
        for i,line in enumerate(filep):
            
            line = preprocessing(line) #preprocessing on data
            language_set.append(line)
          
    return language_set #individual language set

def load_dataset():
    """
        Loading the dataset and generating data X and label y.

    """
    
    X = y = []
    
    for i,l in enumerate(lang):
            
        path = os.path.join(dataset_path,dataset_lang_path[i]) # path to train text
        lang_set = build_dataset(path,l) # generating dataset for individual language
        
        X= X + lang_set # final list of data
        
        y_lang = [l] * len(lang_set) # language label in dataset 
        
        y = y + y_lang # final set of labels
        
    return X, y # Dataset and labels



 