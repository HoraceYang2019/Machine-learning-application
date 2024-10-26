import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Fetch the LFW people dataset
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

# Limit the number of images per person to 50
mask = np.zeros(people.target.shape, dtype=np.bool_)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

# Prepare dataset and normalize pixel values
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.  # Normalize pixel values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

# Step 1: Apply PCA for dimensionality reduction (try with 150 or 200 components)
pca = PCA(n_components=150, whiten=True, random_state=0)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Step 2: Standardize the data after PCA transformation
scaler = StandardScaler()
X_train_pca = scaler.fit_transform(X_train_pca)
X_test_pca = scaler.transform(X_test_pca)

# Step 3: Train an SVM classifier on the PCA-transformed data
svm = SVC(kernel='linear')  # You can experiment with 'rbf' for nonlinear SVMs
svm.fit(X_train_pca, y_train)

# Step 4: Evaluate the classifier on the test data
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)

print("Test set accuracy of SVM after PCA: {:.2f}".format(accuracy))

# Optional: Display a classification report
print(classification_report(y_test, y_pred, target_names=people.target_names))

# Step 5: Visualize the cumulative explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of PCA Components')
plt.show()
