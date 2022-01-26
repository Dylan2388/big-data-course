# Music Genre Prediction Using Acoustic Features: Million Song & LastFM datasets
## Project submission for Managing Big Data, 2022
### A project by Dylan Pham, Raef Kazi, Kristen Phan, and Silvi Fitria

## **Order of running**
In order to replicate the results, the following files must be run in order:
1. lastfm_preprocessing.py 
> This file performs the preprocessing for the LastFM dataset
2. msd_tagged.py
> Joining the Million Song Dataset and the preprocessed LastFM dataset based on their track_id
3. tags_columns_preprocess.py
> Creating columns for the tags to be predicted and preprocessing the numerical data
4. segments_tatums_preprocess.py
> Preprocessing the time series data
5. main.py
> Training various machine learning models and evaluating results
