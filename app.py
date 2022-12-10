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

  
#### streamlit ####

### Title Opening Block
st.title("Machine Learning App - LinkedIn Status")
## Add header to Describe app
st.markdown("Could you be a LinkedIn User?")

### User input
name = st.text_input("Please enter your name","")
st.write(f'Hello {name}! Welcome to  the Machine Learning Application designed by Ali Mohamed. This application was specifically designed to predict if a person is a LinkedIn user.  Please complete the survey below to get a prediction.')

###Age
### Slider
st.write(f'{name}, please start the survey by entering the age of the person:')
a = st.slider('Select an number', 1, 97, 1)

###Female
### Select box"
f = st.selectbox(label="Is the person a woman?",
options=("Yes", "No"))

#Parent
### Select box"
p = st.selectbox(label="Is this person a parent?",
options=("Yes", "No"))

#Married
### Select box"
m = st.selectbox(label="Is this person married?",
options=("Yes", "No"))


#Income
### Slider
with st.container():
        st.write(f'{name}, please select household income from 1-9, that corresponds with income level:')
        st.write(pd.DataFrame({
        'income': [1,2,3,4,5,6,7,8,9],
        'household': ['Less than $10,000','10 to under $20,000','20 to under $30,000','30 to under $40,000','40 to under $50,000','50 to under $75,000','75 to under $100,000','100 to under $150,000','$150,000 or more'],      
        }).to_spring(index=False))
        i = st.slider('Select an number below:',1,9, 1)

#Education
### Slider
with st.container():
        st.write(f'{name}, please select highest level of school/degree completed from the educaiton level below:')
        st.write(pd.DataFrame({
        'Education Level': [1,2,3,4,5,6,7,8],
        'Degree Completed': ['Less than high school','High school incomplete','High school graduate','Some college, no degree','Two-year associate degree from a college or university','Four-year college or university degree',
        'Some postgraduate or professional schooling, no postgraduate degree','Postgraduate or professional degree'],      
        }).to_spring(index=False))
        e = st.slider('Select an number below:',1,8, 1)

st.write(a,f,p,m,i,e)