import os
import cv2
import numpy as np
from skimage import feature
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import argparse

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("-f", action="store", help="Path to an image with an unknown face", required=True)
args = parser.parse_args()


DATASET_PATH = 'training'
IMAGE_SIZE = (128, 128)

def load_images_and_labels(dataset_path):
    images = []
    labels = []
    
    for label in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, label)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, IMAGE_SIZE)  
                images.append(image)
                labels.append(label)
    
    return images, labels

def extract_features(images):
    lbp_features = []
    hog_features = []
    
    for image in images:
        lbp = feature.local_binary_pattern(image, P=8, R=1, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        hist = hist.astype("float")
        hist /= hist.sum()  

        lbp_features.append(hist)

        hog_feature = hog(image, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False)
        hog_features.append(hog_feature)
    
    return np.array(lbp_features), np.array(hog_features)


images, labels = load_images_and_labels(DATASET_PATH)

lbp_features, hog_features = extract_features(images)

combined_features = np.hstack((lbp_features, hog_features))

# X train is 2D array of features
# Y train is for labels
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.3, random_state=42, train_size=0.7)

# simple vector classification
model = SVC(kernel='linear', probability=True)

# will train the model based on ABOVE extracted data
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('===')
# print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='weighted')  # or 'macro' or 'micro'
print(f'F1 Score: {f1}')


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, IMAGE_SIZE)
    
    lbp = feature.local_binary_pattern(image, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= hist.sum()
    hog_feature = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False)
    
    combined_feature = np.hstack((hist, hog_feature))
    
    return combined_feature.reshape(1, -1)

def test_image(model, image_path):
    processed_image = preprocess_image(image_path)
    
    prediction = model.predict(processed_image)
    print('===')
    print(f'The predicted label for the image is: {prediction[0]}')

test_image_path = args.f
test_image(model, test_image_path)
