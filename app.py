import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r'C:\Users\user\Documents\school\Visual Code\Diabetes\cv fill\diabetes.csv')

# Headings
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# Splitting the data
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Function to collect user input
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 67.0, 20.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_data, index=[0])
    return report_data

# Get user data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)
user_result = model.predict(user_data)

# Visualization section
st.title('Visualised Patient Report')

color = 'red' if user_result[0] == 1 else 'blue'

def plot_feature(feature):
    fig = plt.figure()
    sns.scatterplot(x='Age', y=feature, data=df, hue='Outcome', palette='coolwarm')
    sns.scatterplot(x=user_data['Age'], y=user_data[feature], s=200, color=color)
    plt.title('0 - Healthy & 1 - Diabetic')
    st.pyplot(fig)

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction']

for feature in features:
    st.header(f'{feature} vs Age (Your Value vs Dataset)')
    plot_feature(feature)

# Final report
st.subheader('Your Report:')
if user_result[0] == 0:
    st.success('You are not Diabetic.')
else:
    st.error('You are Diabetic.')

st.subheader('Model Accuracy:')
st.write(f'{accuracy_score(y_test, model.predict(X_test))*100:.2f}%')





