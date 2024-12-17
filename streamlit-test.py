import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title("Exploratory Data Analysis (EDA)")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Overview")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write("Column Names:")
    st.write(df.columns)
    
    st.subheader("Preview of the Data")
    st.write(df.head())
    
    st.subheader("Dataset Summary")
    st.write(df.describe())
    
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])
    
    st.subheader("Distribution of Numerical Features")
    num_columns = df.select_dtypes(include=[np.number]).columns
    if len(num_columns) > 0:
        for col in num_columns:
            st.write(f"Distribution for {col}")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
    
    st.subheader("Correlation Heatmap")
    if len(num_columns) > 1:
        corr = df[num_columns].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    
    st.subheader("Pairplot for Numerical Features")
    if len(num_columns) > 1:
        st.write(sns.pairplot(df[num_columns]).figure)
    
    st.subheader("Unique Values in Categorical Features")
    cat_columns = df.select_dtypes(include=[object]).columns
    if len(cat_columns) > 0:
        for col in cat_columns:
            st.write(f"Unique values in {col}: {df[col].nunique()}")
            st.write(df[col].value_counts())
    
    st.subheader("Categorical Features Visualization")
    if len(cat_columns) > 0:
        for col in cat_columns:
            st.write(f"Barplot for {col}")
            fig, ax = plt.subplots()
            sns.countplot(x=df[col], ax=ax)
            st.pyplot(fig)

else:
    st.write("Please upload a CSV file to begin EDA.")
