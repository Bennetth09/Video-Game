from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset globally
data = pd.read_csv('D:\PROJECTS\GAME DEV\data\game_data1.csv')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Game recommendation route
@app.route('/predict', methods=['POST'])
def predict():
    label_encoder = LabelEncoder()
    data['Gen'] = label_encoder.fit_transform(data['Gen'])
    X = data[['Gen', 'Age']]
    y = data[['Game1', 'Game2', 'Game3']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train KNN classifier
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    # Get input from form
    name = request.form['name']
    gen = int(request.form['gender'])
    age = int(request.form['age'])

    # Make a prediction
    sample_input = pd.DataFrame([[gen, age]], columns=['Gen', 'Age'])
    recommendation = classifier.predict(sample_input)

    # Display the result
    game1, game2, game3 = recommendation[0]
    return render_template('result.html', name=name, game1=game1, game2=game2, game3=game3)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
