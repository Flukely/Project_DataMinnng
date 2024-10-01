from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# สร้างแอปพลิเคชัน Flask
app = Flask(__name__)

# โหลดข้อมูลจากไฟล์ Excel
df = pd.read_csv('DataComsci.csv')

# ขั้นตอนที่ 1: เตรียมข้อมูล
# แปลงเกรดเป็นค่าตัวเลข
grade_mapping = {'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0}
df.replace(grade_mapping, inplace=True)

# แทนค่าที่หายไปด้วยค่าเฉลี่ยของแต่ละคอลัมน์สำหรับคอลัมน์ตัวเลข
df.fillna(df.mean(numeric_only=True), inplace=True)

# เลือกฟีเจอร์ที่เกี่ยวข้อง (รายวิชาที่จะใช้ทำนาย GPA)
features = [
        'CalculusforScience', 'FundamentalsOfProgramming', 'HistoryAndDevelopmentOfComputerTechnology',
        'MathematicsForScience', 'ObjectOrientedProgramming', 'ThaiLanguageSkills',
        'PoliticsEconomyandSociety', 'PhilosophyOfScience', 'ManAndEnvironment',
        'LifeSkills', 'LanguageSocietyAndCulture', 'EnglishCriticalReadingForEffectiveCommunication',
        'ComputerArchitecture', 'DataStructure', 'DatabaseSystems',
        'DiscreteMathematicsForComputerScience', 'LinearAlgebraAndApplications',
        'OperatingSystems', 'StatisticalAnalysis', 'EnglishWritingForEffectiveCommunication',
        'AlgorithmDesignandAnalysis', 'ArtificialIntelligence', 'ComputerNetworkAndDataCommunication',
        'Seminar', 'SoftwareEngineering', 'DataMiningTechniques', 
        'MobileApplicationDevelopment', 'MultimediaAnd WebTechnology', 
        'SensingAndActuationForInternetOfThings','SystemAnalysisAndDesign'
]

X = df[features]  # สร้าง DataFrame สำหรับฟีเจอร์

# ตัวแปรเป้าหมาย (GPA)
y = df['GPAgraduate']

# ขั้นตอนที่ 2: แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ขั้นตอนที่ 3: สร้างโมเดล Linear Regression และฝึกสอนมัน
model = LinearRegression()
model.fit(X_train, y_train)

# ขั้นตอนที่ 4: ทำนายโดยใช้ชุดทดสอบ
y_pred = model.predict(X_test)

# ขั้นตอนที่ 5: ประเมินผลโมเดล
mse = mean_squared_error(y_test, y_pred)  # คำนวณ Mean Squared Error
r2 = r2_score(y_test, y_pred)  # คำนวณ R-Squared

# แสดงผลลัพธ์
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-Squared: {r2}')

@app.route('/')
def home():
    return render_template('index.html', features=features)  # ส่งชื่อฟีเจอร์ไปยัง template

@app.route('/predict', methods=['POST'])
def predict():
    grades = []
    for i in range(30):  # จำนวนฟีเจอร์
        grade_key = f'grade{i+1}'
        grades.append(grade_mapping[request.form[grade_key]])

    # สร้าง array ของข้อมูลใหม่ที่ใช้ทำนาย
    new_data = np.array([grades]).reshape(1, -1)
    predicted_gpa = model.predict(new_data)

    # ประเมินผลโมเดลด้วยค่าทำนายล่าสุด
    mse = mean_squared_error(y_test, y_pred)  # ใช้ค่าทำนายชุดทดสอบที่มี
    r2 = r2_score(y_test, y_pred)

    # ส่งค่าผลการทำนาย GPA, MSE, และ R² กลับไปที่ฝั่ง frontend
    return jsonify({
        'predicted_gpa': predicted_gpa[0],
        'mse': mse,
        'r2': r2
    })

if __name__ == '__main__':
    app.run(debug=True)
