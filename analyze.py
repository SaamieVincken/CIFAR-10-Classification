import torchvision
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from setup.data import get_data, get_transform, get_labels


transform = get_transform()
traindata, testdata, trainloader, testloader = get_data(transform, 32)
classes = get_labels()


# Define if classes are balanced
label_counts = Counter(traindata.targets)
print(label_counts)


# Display image by un-normalizing the images
def imshow(img):
    img = img / 2 + 0.5
    numpyimg = img.numpy()
    plt.imshow(np.transpose(numpyimg, (1, 2, 0)))
    plt.show()


# Get random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)


# Display images
imshow(torchvision.utils.make_grid(images))
