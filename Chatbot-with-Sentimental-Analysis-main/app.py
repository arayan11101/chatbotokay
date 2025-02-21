import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# Load your dataset
df = pd.read_csv('C:\\Chatbot-with-Sentimental-Analysis-main\\Chatbot-with-Sentimental-Analysis-main\\Chatbot-with-Sentimental-Analysis-main\\IMDB_Dataset.csv.csv')


# Preprocess the text data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Check if the 'review' column exists in the dataset
if 'review' in df.columns:
    df['cleaned_text'] = df['review'].apply(preprocess_text)
else:
    raise ValueError("The dataset does not contain a 'review' column.")

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Flask app
from flask import Flask, render_template,url_for,redirect
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import InputRequired, Length

class ChatForm(FlaskForm):
    message = StringField('Message', validators=[InputRequired(), Length(max=255)],
                          render_kw={'placeholder': "Enter your message..."})
    submit = SubmitField('Send')
    clear=SubmitField('clear')
    
app = Flask(__name__)
app.config['SECRET_KEY'] = "this is a secret key"

@app.route('/', methods=['GET','POST'])
def chat():
    form=ChatForm()
    response=''
    if form.validate_on_submit():
        user_input = form.message.data
        cleaned_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        sentiment = model.predict(input_vector)[0]
        
        if sentiment == 'positive':
            response = "Positive"
        elif sentiment == 'negative':
            response = "Negative"
        else:
            response = "Neutral"
        
    return render_template('main.html',form=form,response=response)

if __name__ == '__main__':
    app.run(debug=True)

