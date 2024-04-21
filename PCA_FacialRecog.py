import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Function to load images and preprocess them
def load_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100))  # Resize images to a consistent size
        images.append(img.flatten())
    return np.array(images)

# Function to perform PCA and train a KNN classifier
def train_pca_knn(images, labels):
    pca = make_pipeline(StandardScaler(), PCA(n_components=100, whiten=True))
    pca.fit(images)

    X_pca = pca.transform(images)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_pca, labels)

    return pca, knn

# Function to recognize faces in test images
def recognize_faces(test_images, pca, knn):
    test_images_flat = [img.flatten() for img in test_images]
    X_test_pca = pca.transform(test_images_flat)
    predictions = knn.predict(X_test_pca)
    return predictions

# Example usage
if __name__ == "__main__":
    # Load training images and corresponding labels
    # Replace these paths with the paths to your training images and labels
    train_image_paths = ["path/to/train/image1.jpg", "path/to/train/image2.jpg", ...]
    train_labels = [0, 1, ...]  # Labels corresponding to each training image

    train_images = load_images(train_image_paths)

    # Train PCA and KNN classifier
    pca, knn = train_pca_knn(train_images, train_labels)

    # Load test images
    # Replace these paths with the paths to your test images
    test_image_paths = ["path/to/test/image1.jpg", "path/to/test/image2.jpg", ...]
    test_images = load_images(test_image_paths)

    # Recognize faces in test images
    predictions = recognize_faces(test_images, pca, knn)

    # Print predictions
    print("Predictions:", predictions)
