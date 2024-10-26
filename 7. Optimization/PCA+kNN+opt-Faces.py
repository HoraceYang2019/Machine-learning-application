import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Fetch the LFW people dataset
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

# Limit number of images per person to 50
mask = np.zeros(people.target.shape, dtype=np.bool_)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

# Prepare dataset and normalize pixel values
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

# Step 1: Apply PCA for dimensionality reduction (optimize n_components with GridSearchCV)
pca = PCA(whiten=True, random_state=0)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Step 2: Standardize the data after PCA transformation
scaler = StandardScaler()
X_train_pca = scaler.fit_transform(X_train_pca)
X_test_pca = scaler.transform(X_test_pca)

# Step 3: Perform GridSearch to optimize k-NN hyperparameters
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15],  # Test various k-values
    'weights': ['uniform', 'distance'],      # Test weighted k-NN
    'p': [1, 2]                              # Test Manhattan (p=1) and Euclidean (p=2) distances
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_pca, y_train)

# Get the best parameters from grid search
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# Step 4: Evaluate the best k-NN model on test data
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)

print("Test set accuracy after optimization: {:.2f}".format(accuracy))

# Optional: Display a classification report
print(classification_report(y_test, y_pred, target_names=people.target_names))

# Step 5: Visualize the cumulative explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of PCA Components')
plt.show()
