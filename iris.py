import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("iris.data")

# Use required features
X = df[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = df['iris']

# Encode target variable (categorical to numerical)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Logistic Regression model (suitable for classification)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Save the trained model
pickle.dump(classifier, open('modeliris.pkl', 'wb'))

# Save the label encoder to decode predictions later
pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))

'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
input_data = pd.DataFrame([[2.6, 8, 10.1]], columns=['sepal length','sepal width','petal length','petal width'])
print(model.predict(input_data))
'''