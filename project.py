from flask import Flask, render_template, request
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
curr_dir = os.getcwd()
print(curr_dir)

@app.route('/', methods=['POST', "get"])
def reaction():
   
    return render_template('index.html')
@app.route('/show', methods=['post', "get"])
def show():
    if request.method == 'POST' :
                file = request.files['photo'].read()
                if file:
                    npimg = np.fromstring(file, np.uint8)
                    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                    cv2.imwrite(f'{curr_dir}/static/detection.jpg',img)
                    x=1
                    return render_template('index.html', x=x)
                return render_template('index.html')
@app.route('/detect', methods=['post', "get"])
def detect():
    if request.method == 'POST' :
                
                    # or yolov5n - yolov5x6, custom
                    model = torch.hub.load(
                    'ultralytics/yolov5', 'custom', path=f'{curr_dir}/best.pt')
                    img=cv2.imread(f'{curr_dir}/static/detection.jpg')
                    results = model(img)
                    img=np.squeeze(results.render())
                    cv2.imwrite(f'{curr_dir}/static/detection.jpg',img)
                    return render_template('index.html', z=results)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
