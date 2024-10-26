import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
 # min_faces_per_person: The extracted dataset will only retain pictures of people that have at least min_faces_per_person different pictures
image_shape = people.images[0].shape
image_name = people.target_names[people.target[0]]
# count how often each target appears
counts = np.bincount(people.target)

# %%
fig, axes = plt.subplots(2, 5, figsize=(8, 5),
                         subplot_kw={'xticks': (), 'yticks': ()})

for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
plt.show()

#%%
mask = np.zeros(people.target.shape, dtype = np.bool_)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
    
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.
#%%
from sklearn.neighbors import KNeighborsClassifier
# split the data in training and test set
people_X_train, people_X_test, people_y_train, people_y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)

#%%
people_pca = PCA(n_components=100, whiten=True, random_state=0).fit(people_X_train)
people_X_train_pca = people_pca.transform(people_X_train)
people_X_test_pca = people_pca.transform(people_X_test)

fig, axes = plt.subplots(3, 5, figsize=(8, 5),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(people_pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape),
              cmap='viridis')
    ax.set_title("{}. component".format((i + 1)))
plt.show()

# %%
