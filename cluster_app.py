import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel("train.xlsx")

X = df.drop(columns='target', axis=1)
Y = df['target']

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)


def predict(data_point):
    cluster = kmeans.predict(data_point)[0]
    if cluster == 0:
        cluster_type = "A"
    elif cluster == 1:
        cluster_type = "B"
    else:
        cluster_type = "Unknown"
    centre = kmeans.cluster_centers_[cluster]
    explain = f"Data point belongs to cluster {cluster_type} and cluster center is : \n{centre}"
    return cluster_type, explain


st.title('KMeans Clustering Prediction')

input_data_str = st.text_input("Enter data points separated by commas : ")

if input_data_str:
    input_data = np.array([float(x.strip()) for x in input_data_str.split(",")])
    input_data = input_data.reshape(1, -1)

    if len(input_data[0]) == X.shape[1]:
        predicted_cluster, explanation = predict(input_data)
        st.write("Predicted Cluster :", predicted_cluster)
        st.write("Explanation :", explanation)
    else:
        st.write("Number of input data points doesn't match with the number of features.")
