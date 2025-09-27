import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

# --- STEP 1: SETUP ---
print("--- Step 1: Setup ---")
dataset_path = 'train'
if not os.path.exists(dataset_path):
    print(f"Error: The path '{dataset_path}' does not exist.")
    exit()
else:
    print("Dataset path found. Continuing...")


# --- STEP 2: LOAD AND PREPARE IMAGE DATA ---
print("\n--- Step 2: Loading and Preparing Image Data ---")
IMAGE_LIMIT = 2000
IMAGE_SIZE = (64, 64)

images = []
labels = []
all_filenames = os.listdir(dataset_path)
random.shuffle(all_filenames)
filenames = all_filenames[:IMAGE_LIMIT]
print(f"Loading {len(filenames)} images...")

for i, filename in enumerate(filenames):
    if filename.startswith('cat'):
        label = 0
    elif filename.startswith('dog'):
        label = 1
    else:
        continue
    image_path = os.path.join(dataset_path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = cv2.resize(image, IMAGE_SIZE)
        images.append(image)
        labels.append(label)

print("\nImage loading and preparation complete.")


# --- STEP 3: SPLIT DATA AND TRAIN SVM MODEL ---
print("\n--- Step 3: Splitting Data and Training Model ---")
images = np.array(images)
labels = np.array(labels)
n_samples = len(images)
data = images.reshape(n_samples, -1)
data = data / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, random_state=42
)

print(f"Training set has {len(X_train)} samples.")
print(f"Testing set has {len(X_test)} samples.")

print("\nTraining the SVM model...")
model = svm.SVC(kernel='linear', C=1)
model.fit(X_train, y_train)
print("Model training complete.")


# --- STEP 4: EVALUATE MODEL PERFORMANCE ---
print("\n--- Step 4: Evaluating Model Performance ---")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print a detailed classification report
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# Display some example predictions
print("\nDisplaying some test predictions...")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
class_names = ['Cat', 'Dog']

for i, ax in enumerate(axes.flat):
    # Reshape the flattened data back to a 64x64 image
    image = X_test[i].reshape(IMAGE_SIZE)
    ax.imshow(image, cmap='gray')

    true_label = class_names[y_test[i]]
    predicted_label = class_names[y_pred[i]]
    
    # Set title color to green for correct predictions, red for incorrect
    color = "green" if predicted_label == true_label else "red"
    ax.set_title(f"True: {true_label}\nPred: {predicted_label}", color=color)
    ax.axis('off')

plt.tight_layout()
plt.show()