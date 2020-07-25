# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Implementation of a Language detection model using Naive Bayes Classifier

Constants
"""
import os

#We consider 5 languages from the Corpora dataset (Add new languages in this list)
lang = ["french" ,"english","german","italian","dutch"]

#Dataset path (Update correct path )
dataset_path = "Data"
dataset_lang_path = ["fra_newscrawl_2014_10K/fra_newscrawl_2014_10K-sentences.txt","eng_wikipedia_2016_10K/eng_wikipedia_2016_10K-sentences.txt","deu_newscrawl_2017_10K/deu_newscrawl_2017_10K-sentences.txt","ita_wikipedia_2016_10K/ita_wikipedia_2016_10K-sentences.txt","nld_wikipedia_2016_10K/nld_wikipedia_2016_10K-sentences.txt"]

#path to store the model checkpoints 
path = "checkpoints" 
checkpoint_path = os.path.join(path,'naive_bayes_classifier.pickle')