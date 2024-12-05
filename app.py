from flask import Flask, request, render_template, send_file, session
import os
import pdfplumber
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import random
import re
import phonenumbers
import csv
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key for session management

# Load your trained model and tokenizer
model = load_model('model/resume_parser_model.h5')
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts([])  # Fit on the texts used for training

# Define a mapping from keywords to job titles
keyword_to_label = {
    'manager': 'Manager',
    'sales manager': 'Sales Manager',
    'project manager': 'Project Manager',
    'operations manager': 'Operations Manager',
    'analyst': 'Analyst',
    'data analyst': 'Data Analyst',
    'business analyst': 'Business Analyst',
    'software engineer': 'Software Engineer',
    'civil engineer': 'Civil Engineer',
    'mechanical engineer': 'Mechanical Engineer',
    'developer': 'Developer',
    'web developer': 'Web Developer',
    'app developer': 'App Developer',
    'sales': 'Sales',
    'sales executive': 'Sales Executive',
    'customer service': 'Customer Service',
    'marketing': 'Marketing',
    'marketing manager': 'Marketing Manager',
    'technician': 'Technician',
    'help desk technician': 'Help Desk Technician',
    'consultant': 'Consultant',
    'financial analyst': 'Financial Analyst',
    'director': 'Director',
    'executive': 'Executive',
    'administrative assistant': 'Administrative Assistant',
    'research analyst': 'Research Analyst',
    'human resources': 'Human Resources',
    'HR manager': 'HR Manager',
    'accountant': 'Accountant',
    'software developer': 'Software Developer',
    'quality assurance': 'Quality Assurance',
    'IT specialist': 'IT Specialist',
    'nurse': 'Nurse',
    'teacher': 'Teacher',
    'graphic designer': 'Graphic Designer',
    'data scientist': 'Data Scientist',
    'project coordinator': 'Project Coordinator',
    'program manager': 'Program Manager',
    'executive assistant': 'Executive Assistant',
}

# Define a list of skills to search for
skills_keywords = [
    "Python", "Machine Learning", "Data Analysis", "Project Management",
    "Java", "C++", "JavaScript", "SQL", "HTML", "CSS", "React",
    "Node.js", "Django", "Flask", "Excel", "PowerPoint", "Communication",
    "Teamwork", "Leadership", "Problem-Solving", "Agile", "Scrum"
]

# Function to clean the text
def clean_text(text):
    return text.strip()  # Example cleaning logic

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload():
    # Check if the 'resume' part is in the request
    if 'resume' not in request.files:
        return render_template('index.html', error="No file part.")

    file = request.files['resume']

    # Check if a file was selected
    if file.filename == '':
        return render_template('index.html', error="No selected file.")

    # Validate the file format
    if file and file.filename.endswith('.pdf'):
        # Save the uploaded PDF file to a folder
        upload_folder = 'uploads/'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Extract text and parse the resume
        text = extract_text_from_pdf(file_path)
        cleaned_text = clean_text(text)

        # Check if the text extraction was successful
        if not cleaned_text:
            return render_template('index.html', error="Failed to extract text from the uploaded PDF.")

        # Prepare the text for prediction
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=100)

        # Predict using the model
        prediction = model.predict(padded_sequence)

        # Extract information
        education = extract_education(cleaned_text)
        skills = extract_skills(cleaned_text)
        key_info = extract_key_information(cleaned_text)

        # Store results in a structured format
        results = [{
            'filename': file.filename,
            'name': key_info['name'],
            'email': key_info['email'],
            'phone': key_info['phone'],
            'job_title': key_info['job_title'],
            'skills': skills,
            'education': education,
        }]
        
        # Store results in the session for later use
        session['results'] = results

        # Render results on the index page
        return render_template('index.html', results=results)

    else:
        return render_template('index.html', error="Invalid file format. Please upload a PDF file.")


def extract_job_title(text):
    for keyword, title in keyword_to_label.items():
        if keyword in text.lower():
            return title
    return "Job title not found."

def extract_key_information(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'(\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}'
    phone_matches = re.findall(phone_pattern, text)

    email = re.findall(email_pattern, text)
    name = text.split('\n')[0] if text else 'N/A'
    job_title = extract_job_title(text)

    # Normalize phone numbers
    normalized_phone = 'N/A'
    if phone_matches:
        for match in phone_matches:
            try:
                phone_number = phonenumbers.parse(match, "US")  # Adjust region code if necessary
                if phonenumbers.is_valid_number(phone_number):
                    normalized_phone = phonenumbers.format_number(phone_number, phonenumbers.PhoneNumberFormat.E164)
                    break  # Get the first valid number
            except phonenumbers.NumberParseException:
                continue

    return {
        'name': name,
        'email': email[0] if email else 'N/A',
        'phone': normalized_phone,
        'job_title': job_title
    }

def extract_skills(text):
    extracted_skills = [skill for skill in skills_keywords if skill.lower() in text.lower()]
    return ", ".join(extracted_skills) if extracted_skills else "No skills found."

def extract_education(text):
    education_keywords = ["Bachelor", "Master", "PhD", "B.Sc", "M.Sc", "University", "College"]
    extracted_education = [edu for edu in education_keywords if edu.lower() in text.lower()]
    return ", ".join(extracted_education) if extracted_education else "No education information found."



@app.route('/parse', methods=['POST'])
def parse():
    if 'resume' not in request.files:
        return "No file part"
    
    file = request.files['resume']
    
    if request.form['option'] == 'upload':
        if file.filename == '':
            return "No selected file"
        
        if file and file.filename.endswith('.pdf'):
            upload_folder = 'uploads/'
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Extract text and parse the resume
            text = extract_text_from_pdf(file_path)
            cleaned_text = clean_text(text)

            if not cleaned_text:
                return "Failed to extract text from the uploaded PDF."
            
            sequence = tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=100)

            # Predict using the model
            prediction = model.predict(padded_sequence)
            education = extract_education(cleaned_text)
            skills = extract_skills(cleaned_text)
            key_info = extract_key_information(cleaned_text)

            results = [{
                'filename': file.filename,
                'name': key_info['name'],
                'email': key_info['email'],
                'phone': key_info['phone'],
                'job_title': key_info['job_title'],
                'skills': skills,
                'education': education,
            }]
            
            session['results'] = results
            return render_template('index.html', results=results)

        else:
            return "Invalid file format. Please upload a PDF file."
    
    elif request.form['option'] == 'database':
        resumes_folder = 'resume/'
        all_pdfs = [f for f in os.listdir(resumes_folder) if f.endswith('.pdf')]
        
        sample_size = min(20, len(all_pdfs))
        selected_pdfs = random.sample(all_pdfs, sample_size)
        results = []

        print("Starting PDF parsing...")

        for pdf_file in selected_pdfs:
            pdf_path = os.path.join(resumes_folder, pdf_file)
            print(f'Extracting text from {pdf_file}...')

            text = extract_text_from_pdf(pdf_path)
            cleaned_text = clean_text(text)

            if not cleaned_text:
                print(f"Failed to extract text from {pdf_file}.")
                continue

            sequence = tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=100)

            prediction = model.predict(padded_sequence)

            education = extract_education(cleaned_text)
            skills = extract_skills(cleaned_text)
            key_info = extract_key_information(cleaned_text)

            results.append({
                'filename': pdf_file,
                'name': key_info['name'],
                'email': key_info['email'],
                'phone': key_info['phone'],
                'job_title': key_info['job_title'],
                'skills': skills,
                'education': education,
            })
        
        print("Completed PDF parsing.")
        session['results'] = results
        return render_template('index.html', results=results)
    
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    if not query:
        return redirect(url_for('index'))  # Redirect if no query is provided

    results = []

    # Assuming you have a list of results stored in session
    if 'results' in session:
        for resume in session['results']:
            # Check if query matches any of the desired fields (case insensitive)
            if (query.lower() in resume['job_title'].lower() or
                query.lower() in resume['skills'].lower() or
                query.lower() in resume['education'].lower()):
                results.append(resume)

    return render_template('index.html', results=results)


    
# Route to download results as CSV
@app.route('/download/csv', methods=['GET'])
def download_csv():
    results = session.get('results', [])
    output_file = 'parsed_resumes.csv'
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Name', 'Email', 'Phone', 'Job Title', 'Skills', 'Education'])
        for result in results:
            writer.writerow([result['filename'], result['name'], result['email'], result['phone'],
                             result['job_title'], result['skills'], result['education']])
    
    return send_file(output_file, as_attachment=True)



# Route to download results as JSON
@app.route('/download/json', methods=['GET'])
def download_json():
    results = session.get('results', [])
    output_file = 'parsed_resumes.json'
    
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)
    
    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
