import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Check vector lengths
lengths = [len(d) for d in data_dict['data']]
unique_lengths = set(lengths)
print("Feature vector lengths:", unique_lengths)

# Remove bad samples if any
if len(unique_lengths) > 1:
    clean_data = []
    clean_labels = []
    for d, l in zip(data_dict['data'], data_dict['labels']):
        if len(d) == 42:
            clean_data.append(d)
            clean_labels.append(l)
    data = np.asarray(clean_data)
    labels = np.asarray(clean_labels)
    print(f"Clean samples kept: {len(data)}")
else:
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

print("Class distribution:", collections.Counter(labels))

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"{score*100:.2f}% of samples were classified correctly!")

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as model.p")
