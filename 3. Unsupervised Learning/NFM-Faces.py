import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape, dtype = np.bool_)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
    
X_people = people.data[mask]
y_people = people.target[mask]
# split the data in training and test set
people_X_train, people_X_test, people_y_train, people_y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)

#%%
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0, max_iter=2000)
nmf.fit(people_X_train)
people_X_train_nmf = nmf.transform(people_X_train)
people_X_test_nmf = nmf.transform(people_X_test)

#fig, axes = plt.subplots(3, 5, figsize=(8, 5),
#                         subplot_kw={'xticks': (), 'yticks': ()})
# for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
#     ax.imshow(component.reshape(image_shape))
#     ax.set_title("Comp. #{}".format(i))
# plt.show()

#%%
compn = 3
# sort by 3rd component, plot first 10 images
inds = np.argsort(people_X_test_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(8, 5),
                         subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("Large component 3")
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(people_X_test[ind].reshape(image_shape))
   # ax.set_title(people.target_names[people.target[ind]])
plt.show()
# %%
