# train_model.py
# Re-trains the synthetic-model used by this project.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# (This file contains the same synthetic-data training pipeline used to produce model.pkl)
np.random.seed(42)
n = 2000
N = np.random.randint(0, 141, size=n)
P = np.random.randint(0, 141, size=n)
K = np.random.randint(0, 141, size=n)
pH = np.round(np.random.uniform(3.5, 9.0, size=n), 2)
rainfall = np.round(np.random.uniform(0, 400, size=n), 1)
temp = np.round(np.random.uniform(5, 40, size=n), 1)

crops = []
for i in range(n):
    if (rainfall[i] > 200 and temp[i] > 20) or (pH[i] > 6.0 and N[i] > 80):
        crops.append("Rice")
    elif (temp[i] < 18 and rainfall[i] < 150) or (pH[i] < 6.0 and K[i] < 50):
        crops.append("Wheat")
    elif (temp[i] > 22 and rainfall[i] < 150 and K[i] > 40):
        crops.append("Maize")
    elif (pH[i] >= 5.5 and pH[i] <= 7.5 and rainfall[i] > 100 and N[i] < 60):
        crops.append("Sugarcane")
    else:
        crops.append("Millet")

X = pd.DataFrame({"N":N,"P":P,"K":K,"pH":pH,"rainfall":rainfall,"temperature":temp})
y = crops
le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
with open("model/model.pkl", "wb") as f:
    pickle.dump({"model": clf, "le": le}, f)
print("Saved model to model/model.pkl")
