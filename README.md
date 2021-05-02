# Deep-Learning-Project

####################################
Dataset
###################################

Each line in the file has the following format:

<tweet> <label>

where
- <tweet> is a sentence of tweet contents
- <label> is a real number between -3 to 3 indicating the level of positive and negative sentiment intensity, that  best represents the mental state of the tweeter.

The classes:
- 3: very positive mental state can be inferred
- 2: moderately positive mental state can be inferred
- 1: slightly positive mental state can be inferred
- 0: neutral or mixed mental state can be inferred
- -1: slightly negative mental state can be inferred
- -2: moderately negative mental state can be inferred
- -3: very negative mental state can be inferred

-------------------------------------------------------------
We provided two types of dataset. The first one is the original tweet sentences, which includes:
 
- train_df.csv
- dev_df.csv

The second one is the cleaned tweet sentences. We exclude special characters, '@user' contents and so on:

- train_df_clean.csv
- dev_df_clean.csv

------------------------------------------------------------
Command to run the deep models:
```
cd scripts
sh sentiment_train_predict.sh
```

There are three different tasks in 'sentiment_train_predict.sh'
1. run training and prediction on V-oc dataset
2. run training on V-oc and EI-oc dataset and predict on V-oc dataset with only one head of sequence classification
3. run training on  V-oc and EI-oc dataset and predict on V-oc dataset with two different heads of sequence classification

To run different task, just uncomment the scripts of the corresponding task!
