from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
import os

from pathlib import Path
from calculator import TfIdfCalculator


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt'}

# Создаем экземпляр класса калькулятора TF-IDF
calculator = TfIdfCalculator()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'  # Нужен для flash сообщений

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            with open(filepath, encoding='utf-8') as f:
                content = f.read()
            results = calculator.calculate_tf_idf([content])
            return render_template('index.html', data=results)
    return render_template('index.html')

if __name__ == '__main__':
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    app.run(debug=True)