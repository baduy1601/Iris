from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Khởi tạo Flask app
app = Flask(__name__)

# Load mô hình và label encoder đã lưu
model = pickle.load(open('modeliris.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Dự đoán
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        # Trả kết quả về giao diện
        return render_template('index.html', prediction_text=f'Kết quả dự đoán: {prediction_label}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Lỗi: {e}')

if __name__ == "__main__":
    app.run(debug=True)
