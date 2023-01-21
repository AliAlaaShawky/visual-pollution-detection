import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Model
# or yolov5n - yolov5x6, custom
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=r"C:\Users\xps\Desktop\Theme#1\best.pt")

# Images
# or file, Path, PIL, OpenCV, numpy, list
img = r"C:\Users\xps\Desktop\Theme#1\01e9cf28b75fb555eb5d6116bd1d4fcd.jpg"

# Inference
results = model(img)

# Results
results = model(img)
img = cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_BGR2RGB)
cv2.imwrite('static/detection.jpg',img)

