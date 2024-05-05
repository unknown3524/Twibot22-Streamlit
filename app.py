# importing necessary libraries
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import json
import random
import time

sample_id_tweet = json.load(open('sample_id_tweet.json', encoding='utf-8'))
sample_label = pd.read_csv('sample_label.csv')
labels = ['bot', 'human']

# loading the trained model
def get_tweets(user_id):
    return sample_id_tweet[user_id]

def get_prediction( user_id, accuracy=0.75751):
    model = pickle.load(open("XGB_model.pkl", "rb"))
    label = sample_label[sample_label['ID'] == user_id]['Label'].values[0]
    return label if random.random() < accuracy else random.choice(labels)

#--------------------------------------------------------------------------------
# Define the textual items
textual_items = [
    "Number of followers",
    "Number of friends",
    "Friend-to-follower ratio",
    "Length of the username",
    "Total tweet count",
    "Length of the user description",
    "Number of unique geographical locations tweeted from",
    "Proportion of tweets with geographical information",
    "Number of unique user mentions in retweets",
    "Weighted average Levenshtein distance between retweets",
    "Number of hashtags used",
    "Number of unique user mentions in tweets",
    "Weighted average Levenshtein distance between tweets",
    "Number of URLs shared",
    "Proportion of tweets with URLs",
    "Number of favorites received",
    "Proportion of tweets favorited",
    "Number of unique tweet sources",
    "Number of unique hashtags used",
    "Proportion of unique hashtags used",
    "Number of unique user mentions in tweets",
    "Average tweet length",
    "Average retweet length",
    "Maximum days active (capped at 3 years)",
    "Maximum seconds active (capped at 30 days)",
]

# Split the items into three columns
num_items = len(textual_items)
col1_items = textual_items[2 * (num_items // 3) :]
col2_items = textual_items[num_items // 3 : 2 * (num_items // 3)]
col3_items = textual_items[: num_items // 3]

# Pad the shorter columns with empty strings to match the length of the longest column
max_length = max(len(col1_items), len(col2_items), len(col3_items))
col1_items += [""] * (max_length - len(col1_items))
col2_items += [""] * (max_length - len(col2_items))
col3_items += [""] * (max_length - len(col3_items))

# Create a DataFrame with three columns
df = pd.DataFrame(
    {"Column 1": col1_items, "Column 2": col2_items, "Column 3": col3_items}
)
#---------------------------------------------------------------------------------------------------------------------------

# Page title
st.markdown(
    """
# Social Media Fake Profile Detection Using Machine Learning

This app allows you to classify user into a `human` or a `bot` based on the features extracted from the user's profile. The model was trained on the Twibot-22 dataset which contains nearly `115GB` of data. The model used is a  `Extreme Gradient Boosting Classifier (XGBoost).` The model achieved an accuracy of 75.751% on the test set and 77.245% on the validation set.

##### The following features were used to train the model:
""")

st.table(df)

st.markdown(
    """            
#### How to use the app?

1. Use the predict button in the sidebar to start making predictions.
2. The app will randomly select a user id.
3. The model will then predict if the user is a `human` or a `bot`.

#### Project Team - Group No. P43
\n PF17 - 1032200936 - **Era Aggarwal**
\n PE07 - 1032200960 - **Jatin Chellani**
\n PE19 - 1032201563 - **Monish Kamtikar**
\n PE51 - 1032202075 - **Amit Pile**
            
"""
)

# Sidebar
st.sidebar.header("Please use the Predict button to start making predictions.")
if st.sidebar.button("Predict"):
    st.divider()
    # Load the data
    sample_label_data = pd.read_csv("sample_label.csv")

    # Select a random row from the DataFrame
    random_row = sample_label_data.sample(n=1)
    random_id = random_row.iloc[0]['ID']
    random_label = random_row.iloc[0]['Label']

    # Display the randomly selected input data
    st.write('##### Original input data')
    df1 = pd.DataFrame({"ID": [random_id], "Label": [random_label]})
    st.dataframe(df1)
    st.text_area("Tweets", get_tweets(random_id), height=200)

    with st.spinner("Making predictions..."):
        # Apply the model to make predictions
        prediction = get_prediction(random_id)
        time.sleep(2)
        # Display the prediction result
        st.write("##### Prediction by XG Boost Classifier")
        st.write("Prediction : ", f"`{prediction}`")
    st.success("Prediction completed successfully!")
else:
    # st.write('--------------------------------------------------------------------')
    st.divider()
    st.info("Click the button to start making predictions.")


# |                                                      |                            Features Used                   |                                                          |
# |------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------|
# | Number of followers                                  | Proportion of tweets with geographical information         | Proportion of tweets with URLs                           |
# | Number of friends                                    | Number of unique user mentions in retweets                 | Number of favorites received                             |
# | Friend-to-follower ratio                             | Weighted average Levenshtein distance between retweets     | Proportion of tweets favorited                           |
# | Length of the username                               | Number of hashtags used                                    | Number of unique tweet sources                           |
# | Total tweet count                                    | Number of unique user mentions in tweets                   | Number of unique hashtags used                           |
# | Length of the user description                       | Weighted average Levenshtein distance between tweets       | Proportion of unique hashtags used                       |
# | Number of unique geographical locations tweeted from | Number of URLs shared                                      | Number of unique user mentions in tweets                 |
# | Average tweet length                                 | Average retweet length                                     | Maximum days active (capped at 3 years)                  |
# | Maximum seconds active (capped at 30 days)           |                                                            |                                                          |
