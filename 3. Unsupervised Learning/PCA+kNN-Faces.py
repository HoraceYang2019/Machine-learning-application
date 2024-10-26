import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Fetch the LFW people dataset
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

# Count occurrences of each target value (each person)
counts = np.bincount(people.target)

# Create a mask to limit the number of images per person to 50
mask = np.zeros(people.target.shape, dtype=np.bool_)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

# Prepare the dataset and normalize the pixel values
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.  # Normalize pixel values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

# Step 1: Apply PCA for dimensionality reduction
pca = PCA(n_components = 50, whiten=True, random_state=0)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Step 2: Train a k-NN classifier on the PCA-transformed data
knn = KNeighborsClassifier(n_neighbors=10)  # Use 1-NN as a baseline
knn.fit(X_train_pca, y_train)

# Step 3: Evaluate the classifier on the test data
y_pred = knn.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)

print("Test set accuracy of k-NN after PCA: {:.2f}".format(accuracy))

# Optional: Display a classification report
# print(classification_report(y_test, y_pred, target_names=people.target_names))

# # Step 4: Visualize the cumulative explained variance
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Explained Variance by Number of PCA Components')
# plt.show()
