import pandas as pd
import streamlit as st


@st.cache
def load_data():
    return pd.read_excel('rawdata.xlsx')


df = load_data()

df['time'] = pd.to_timedelta(df['time'].astype(str))
df['DateTime'] = df['date'] + df['time']
df['Duration'] = df.groupby(['date', 'position'])['DateTime'].diff().fillna(pd.Timedelta(seconds=0))
df = df[df['Duration'] >= pd.Timedelta(seconds=0)]

total = df.groupby(['date', 'position'])['Duration'].sum().unstack().fillna(pd.Timedelta(seconds = 0))
activity_count = df.groupby(['date', 'activity'])['activity'].count().unstack().fillna(0)

st.title('Data Analysis App')

st.subheader('Total Duration by Date and Position')
st.write(total)

st.subheader('Activity Count by Date and Activity')
st.write(activity_count)