import cv2
import matplotlib.pyplot as plt

img = cv2.imread('13000.jpg')
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "EDSR_x2.pb"
sr.readModel(path)
sr.setModel("edsr", 2)
result = sr.upsample(img)
print(result)