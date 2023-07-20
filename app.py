from flask import Flask, render_template, request, redirect, url_for, flash
import subprocess
import os
from test import OCR

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/OCR', methods=['POST'])
def myOCR():
    image = request.files['image']
    image.save('plan.jpg')
    
    output = OCR('plan.jpg')
    
    return render_template('OCR.html', output=output)