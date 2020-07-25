# Automatic Language Detection

We create a automatic language identification model using naive bayes classifier for the following 5 languages: English, French, German, Italian and Dutch. 

## Dataset
Wortschatz Leipzig Corpora Collection ([here](https://wortschatz.uni-leipzig.de/en/download)) is used for training and testing.
We use 10K sentences for each language giving a total of 50K sentences.
The train and test data are randomly split into 80% and 20% sets.

## Requirement
Python 3.6.2 and SciKit Learn 0.19.0

## Running
```bash
python language_detection
```

## Output
The accuracy of the model is **99.59%**. 

## Note
The Dataset is not present in the this folder. Please download the dataset from the link provided above and update the correct directory path for the Dataset in [utils.py](https://github.com/debanjali05/Automatic_Language_Detection/blob/master/utils.py). 

The model checkpoints are saved in "checkpoints" folder. Create this folder and update the directory path in [utils.py](https://github.com/debanjali05/Automatic_Language_Detection/blob/master/utils.py) before running. 

Use can also add/change the languages in the [utils.py](https://github.com/debanjali05/Automatic_Language_Detection/blob/master/utils.py) file.
