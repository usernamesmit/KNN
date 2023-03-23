# K-Nearest Neighbors

import pandas as pd 
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("data/car.data")

pre = preprocessing.LabelEncoder()

buying = pre.fit_transform(list(data['buying']))
maint = pre.fit_transform(list(data['maint']))
door = pre.fit_transform(list(data['door']))
person = pre.fit_transform(list(data['person']))
lug_boot = pre.fit_transform(list(data['lug_boot']))
safety = pre.fit_transform(list(data['safety']))
cls = pre.fit_transform(list(data['class']))

X = list(zip(buying, maint, door, person, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors= 9)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

predict = model.predict(x_test)
names = ['unacc','acc','good','vgood']

for i in range(len(predict)):
    print(f"Predicted: {names[predict[i]]}, Data: {x_test[i]}, Actual: {names[y_test[i]]}")
    n = model.kneighbors([x_test[i]], 3, True)
    print(n)
