# Deep-Learning-Project

####################################
Dataset
###################################

Each line in the file has the following format:

<tweet><label>

where
<tweet> is a sentence of tweet contents
<label> is a real number between -3 to 3 indicating the level of positive and negative sentiment intensity, that  best represents the mental state of the tweeter.

The classes:
3: very positive mental state can be inferred
2: moderately positive mental state can be inferred
1: slightly positive mental state can be inferred
0: neutral or mixed mental state can be inferred
-1: slightly negative mental state can be inferred
-2: moderately negative mental state can be inferred
-3: very negative mental state can be inferred

-------------------------------------------------------------
We provided two types of dataset. The first one is the original tweet sentences, which includes:
 
- train_df.csv
- dev_df.csv

The second one is the cleaned tweet sentences. We exclude special characters, '@user' contents and so on:

- train_df_clean.csv
- dev_df_clean.csv
