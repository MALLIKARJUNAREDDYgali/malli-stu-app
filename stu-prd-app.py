import streamlit  as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder,StandardScaler


def load_model():
          with open("student final model.pkl","rb") as file:
                  model,scaler,le = pickle.load(file)
          return model,scaler,le


def preprocessing_input_data(data, scaler, le):
    if data["Extracurricular Activities"] not in le.classes_:
        le.classes_ = np.append(le.classes_, data["Extracurricular Activities"])  # Add missing value
    data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed


def predict_data(data):
        model,scaler,le = load_model()
        processed_data = preprocessing_input_data(data,scaler,le)
        predicition = model.predict(processed_data)
        return predicition

def main():
        st.title("Malli student performance prediction")
        st.write("enter your data to get a prediction for your performance")
        
        hours_Studied = st.number_input("Hours stuided",min_value=2,max_value=10,value=5)
        previous_Scores = st.number_input("previous score",min_value=40,max_value=100,value=60)
        extracurricular_Activities = st.selectbox("Extra curriculam activities",["Yes","No"])
        sleep_Hours = st.number_input("Sleeping hours",min_value=2,max_value=10,value=5)
        SampleQuestion = st.number_input("No of question paper solved",min_value=2,max_value=10,value=5)

        if st.button("predict your score"):
                user_data={
                        "Hours Studied" : hours_Studied,
                        "Previous Scores" :previous_Scores,
                        "Extracurricular Activities" :extracurricular_Activities,
                        "Sleep Hours":sleep_Hours,
                        "Sample Question Papers Practiced":SampleQuestion
                }
                predicition = predict_data(user_data)
                st.success(f"your predicition result is {predicition}")



if __name__ == "__main__":
        main()