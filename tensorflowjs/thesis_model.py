from flask import Flask, render_template
from flask_cors import CORS


# Khởi tạo Flask
app = Flask(__name__)
cors=CORS(app)

# Hàm xử lý request
@app.route("/", methods=['GET'])
def home_page():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False)