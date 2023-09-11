import numpy as np
import rondo
import rondo.functions as F
import rondo.layers as L
import rondo.utils as U

class VGG16(rondo.Model):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1_1 = L.Conv2d(64, kernel=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(1000)

        if pretrained:
            path = U.download('https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz')
            self.load(path)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling_max_2d(x, kernel=2, stride=2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling_max_2d(x, kernel=2, stride=2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling_max_2d(x, kernel=2, stride=2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling_max_2d(x, kernel=2, stride=2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling_max_2d(x, kernel=2, stride=2)
        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        image = image.convert('RGB')
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=dtype) # (H, W, C)

        # Reverse the channels from RGB to BGR
        image = image[:, :, ::-1]
        # Subtract mean values from the BGR channels respectively
        image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
        # Transpose the image array from (H, W, C) to (C, H, W)
        image = image.transpose((2, 0, 1))

        return image
