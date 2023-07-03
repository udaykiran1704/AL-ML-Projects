import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from flask import Flask, request, jsonify

# Step 1: Data Exploration and Preprocessing
df = pd.read_csv('datawine.csv')

# Handle missing values
df = df.dropna(subset=['variety'])

# Convert categorical variables to numerical
df['country'] = df['country'].astype('category').cat.codes
df['province'] = df['province'].astype('category').cat.codes

# Step 2: Actionable Insights

# a. Identify the countries with the highest number of reviews
top_countries = df['country'].value_counts().head(5)
print("Top 5 countries with the most reviews:")
print(top_countries)

# b. Analyze the relationship between price and points
plt.scatter(df['price'], df['points'])
plt.xlabel('Price')
plt.ylabel('Points')
plt.show()

# c. Discover popular wine varieties
top_varieties = df['variety'].value_counts().head(5)
print("Top 5 most popular wine varieties:")
print(top_varieties)

# d. Identify the most influential reviewers
reviewers_avg_ratings = df.groupby('user_name')['points'].mean()
top_reviewers = reviewers_avg_ratings.nlargest(5)
print("Top 5 influential reviewers:")
print(top_reviewers)

# e. Explore the relationship between province and variety
heatmap_data = df.pivot_table(index='province', columns='variety', aggfunc='size')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='Blues')
plt.xlabel('Wine Variety')
plt.ylabel('Province')
plt.show()

# Step 3: Predictive Model for Wine Variety Prediction
X = df[['country', 'points', 'price', 'province']]
y = df['variety']

# Handle missing values using imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_classifier.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 4: Building API for Serving Predictions
# (Assuming usage of Flask for building the API)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_wine_variety():
    data = request.json
    input_data = pd.DataFrame(data)
    input_data['country'] = input_data['country'].astype('category').cat.codes
    input_data['province'] = input_data['province'].astype('category').cat.codes

    # Handle missing values using imputation
    input_data = imputer.transform(input_data)

    prediction = rf_classifier.predict(input_data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()

