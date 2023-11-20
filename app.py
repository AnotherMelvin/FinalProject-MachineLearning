from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    user_data = request.form
    
    bmi = float(user_data['weight']) / ((float(user_data['height'])/100) ** 2)
    
    gender_key = "Gender_" + user_data['gender']
    occupation_key = "Occupation_" + user_data['occup']
    
    if (bmi < 18.5):
        bmi_cat = "Underweight"
    elif (bmi >= 18.5 and bmi < 25):
        bmi_cat_normal1 = "Normal"
        bmi_cat_normal2 = "Normal Weight"
    elif (bmi >= 25 and bmi < 30):
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"
    
    data_value = {
        'Age': [int(user_data['age'])], 
        'Sleep Duration': [float(user_data['sleep'])], 
        'Quality of Sleep': [int(user_data['quality'])], 
        'Physical Activity Level': [float(user_data['physical'])],
        'Stress Level': [int(user_data['stress'])], 
        'Heart Rate': [int(user_data['heart'])], 
        'Daily Steps': [int(user_data['steps'])], 
        'Gender_Female': [False],
        'Gender_Male': [False], 
        'Occupation_Accountant': [False], 
        'Occupation_Doctor': [False],
        'Occupation_Engineer': [False], 
        'Occupation_Lawyer': [False], 
        'Occupation_Manager': [False],
        'Occupation_Nurse': [False], 
        'Occupation_Sales Representative': [False],
        'Occupation_Salesperson': [False], 
        'Occupation_Scientist': [False],
        'Occupation_Software Engineer': [False], 
        'Occupation_Teacher': [False],
        'BMI Category_Normal': [False], 
        'BMI Category_Normal Weight': [False],
        'BMI Category_Obese': [False], 
        'BMI Category_Overweight': [False], 
        'Systolic_BP': [int(user_data['sys'])],
        'Diastolic_BP': [int(user_data['dias'])]
    }
    
    data_point = pd.DataFrame(data_value)
    data_point[gender_key] = True
    
    if (occupation_key != "Occupation_Student" and occupation_key != "Occupation_Other"):
        data_point[occupation_key] = True
    
    if (bmi >= 18.5 and bmi < 25):
        bmi_key1 = "BMI Category_" + bmi_cat_normal1
        bmi_key2 = "BMI Category_" + bmi_cat_normal2
        data_point[bmi_key1] = True
        data_point[bmi_key2] = True
    else:
        bmi_key = "BMI Category_" + bmi_cat
        data_point[bmi_key] = True
    
    prediction, accuracy = predict(data_point)
    
    return render_template('result.html', predict = prediction, acc = accuracy)

def predict(data_point):
    file_path = "data/Sleep_health_and_lifestyle_dataset.csv"
    df = pd.read_csv(file_path)
    
    df = df.drop("Person ID", axis=1)
    df['Sleep Disorder'] = df[['Sleep Disorder']].fillna('None')
    
    df = pd.get_dummies(df, columns=["Gender", "Occupation", "BMI Category"])

    df["Systolic_BP"] = df["Blood Pressure"].apply(lambda x: int(x.split("/")[0]))
    df["Diastolic_BP"] = df["Blood Pressure"].apply(lambda x: int(x.split("/")[1]))
    df = df.drop("Blood Pressure", axis=1)

    X = df.drop("Sleep Disorder", axis=1)
    y = df["Sleep Disorder"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 2) * 100
    
    prediction = model.predict(data_point)
    return prediction[0], accuracy

if __name__ == "__main__":
    app.run(debug=True)