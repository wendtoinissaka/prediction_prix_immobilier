# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:04:21 2023

@author: PC.054
"""

import pandas as pd 
import numpy as np 
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import r2_score

df=pd.read_csv("Housing.csv")

st.sidebar.title("Summary")

pages = ["Project Context", "Data Exploration", "Data Analysis", "Modelling"]

page = st.sidebar.radio("Go to", pages)

if page == pages[0] : 
    
    st.write("### Project Context")
    
    st.write("This project is part of a property context. The aim is to predict the price of a home based on its characteristics, within a financial estimation framework.")
    
    st.write("We have at our disposal the housing.csv file containing properties data. Each row observation corresponds to a housing unit. Each column variable is a housing characteristic.")
    
    st.write("First, we'll explore this dataset. Then we'll analyze it visually to extract information according to certain study axes. Finally, we will implement Machine Learning models to predict the price.")
    
    st.image("immobilier1.png")
    
elif page == pages[1]:
    st.write("### Data Exploration")
    
    st.dataframe(df.head())
    
    st.write("Dimensions of the dataframe :")
    
    st.write(df.shape)
    
    if st.checkbox("Show the missing values") : 
        st.dataframe(df.isna().sum())
        
    if st.checkbox("Show the duplicates") : 
        st.write(df.duplicated().sum())
        
elif page == pages[2]:
    st.write("### Data Analysis")
    
    fig = sns.displot(x='price', data=df, kde=True)
    plt.title("Distribution of the target variable price")
    st.pyplot(fig)
    
    fig2 = px.scatter(df, x="price", y="area", title="Price depending on the area")
    st.plotly_chart(fig2)
    
    fig3, ax = plt.subplots()
    df_numeric = df.select_dtypes(include=[np.number])  # Sélectionne uniquement les colonnes numériques
    sns.heatmap(df_numeric.corr(), ax=ax)
    # sns.heatmap(df.corr(), ax=ax)
    plt.title("Correlation matrix of the variables")
    st.write(fig3)
    
elif page == pages[3]:
    st.write("### Modelling")
    
    df_prep = pd.read_csv("df_preprocessed.csv")
    
    y = df_prep["price"]
    X= df_prep.drop("price", axis=1)
    
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler()
    num = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    X_train[num] = scaler.fit_transform(X_train[num])
    X_test[num] = scaler.transform(X_test[num])
    
    reg = joblib.load("model_reg_line")
    rf = joblib.load("model_reg_rf")
    knn = joblib.load("model_reg_knn")
    
    y_pred_reg=reg.predict(X_test)
    y_pred_rf=rf.predict(X_test)
    y_pred_knn=knn.predict(X_test)
    
    model_chosen = st.selectbox(label = "Model", options = ['Linear Regression', 'Random Forest', 'KNN'])
    
    def train_model(model_chosen) : 
        if model_chosen == 'Linear Regression' :
            y_pred = y_pred_reg
        elif model_chosen == 'Random Forest' :
            y_pred = y_pred_rf
        elif model_chosen == 'KNN' :
            y_pred = y_pred_knn
        r2 = r2_score(y_test, y_pred)
        return r2
    
    st.write("Coefficient of determination", train_model(model_chosen))