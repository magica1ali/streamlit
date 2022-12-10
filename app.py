##Import Libraries
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

## Read and Clean Data 
s = pd.read_csv("social_media_usage.csv")

# define function
def clean_sm(x):
    x=np.where(x == 1,
            1,
            0)
    return(x)

#clean data 
ss = pd.DataFrame({
    "income":np.where(s['income'] > 9, np.nan,s['income']),
    "education":np.where(s['educ2'] > 8, np.nan,s['educ2']),
    "parent":np.where(s['par'] >2, np.nan,
            np.where(s['par'] <=1, 1, 0)),        
    "married":np.where(s['marital'] > 6, np.nan,
            np.where(s['marital'] == 1, 1,0)),
    "female":np.where(s['gender'] >3, np.nan,
             np.where(s['gender'] == 2, 1,0)),
     "age":np.where(s['age'] > 97, np.nan,s['age']),
        "sm_li":clean_sm(s.web1h)})

ss = ss.dropna()

## Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[['income','education','parent','married','female','age']]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y, 
                                                    test_size=.20, 
                                                    random_state=525) 

# Initialize algorithm and set class_weight to balanced
lr = LogisticRegression(class_weight='balanced')

# Fit algorithm to training data
lr.fit(X_train.values,y_train)

### streamlit 
## Add header to Describe app
st.markdown("#Could you be a LinkedIn User?")

# "### Slider"
x = st.slider("x")

######## 4 Add interactive dropdown selection box

### Select box"
answer = st.selectbox(label="Are you a parent",
options=("Yes", "No"))
st.write("Here are some resources for ", answer)