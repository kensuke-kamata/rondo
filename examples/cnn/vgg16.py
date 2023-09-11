import numpy as np
from PIL import Image
import rondo
from rondo.models import VGG16

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
path = rondo.utils.download(url)
img = Image.open(path)

x = VGG16.preprocess(img)
x = x[np.newaxis] # Add batch axis

model = VGG16(pretrained=True)
with rondo.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file='vgg16.pdf')
labels = rondo.datasets.ImageNet.labels()
print(labels[predict_id])
