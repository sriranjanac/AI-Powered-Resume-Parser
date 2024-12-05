# Import necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
import pdfplumber
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk
nltk.download('stopwords')


# Step 1: Extract Text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Specify the folder containing your resumes
pdf_folder = r'"C:\\Users\\gcboo\\OneDrive\\Desktop\\SEMESTER 3\\Fundamentals of AI\\flask\\resume"'  # Replace with your folder path
resume_texts = []
for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, filename)
        text = extract_text_from_pdf(pdf_path)
        resume_texts.append({'filename': filename, 'text': text})

# Convert to DataFrame
df = pd.DataFrame(resume_texts)

# Step 2: Data Preprocessing
# Function to clean and preprocess text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

# Apply the cleaning function
df['cleaned_text'] = df['text'].apply(clean_text)

# Tokenization and stopword removal
stop_words = set(stopwords.words('english'))
df['tokens'] = df['cleaned_text'].apply(lambda x: [word for word in x.split() if word not in stop_words])

# Stemming
ps = PorterStemmer()
df['stemmed'] = df['tokens'].apply(lambda x: [ps.stem(word) for word in x])

# Define a mapping from keywords to labels
keyword_to_label = {
    'manager': 'Manager',
    'sales manager': 'Sales Manager',
    'project manager': 'Project Manager',
    'analyst': 'Analyst',
    'software engineer': 'Software Engineer',
    # Add more keywords and labels as necessary
}

# Function to assign labels based on keyword mapping
def assign_label(cleaned_text):
    cleaned_text = cleaned_text.lower()
    for keyword, label in keyword_to_label.items():
        if keyword in cleaned_text:
            return label
    return 'Other'  # Default label if no keyword matches

# Create the label column
df['label'] = df['cleaned_text'].apply(assign_label)

# Step 3: Prepare data for training
X = df['cleaned_text']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
max_len = max(len(x.split()) for x in X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Encode labels
y_train_encoded = pd.get_dummies(y_train).values
y_test_encoded = pd.get_dummies(y_test).values

# Step 4: Model Training
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(units=y_train_encoded.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_pad, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('resume_parser_model.h5')
