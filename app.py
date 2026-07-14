from flask import Flask, render_template, request
import pdfplumber
import pandas as pd
import joblib
import os

from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -----------------------------
# Create uploads folder
# -----------------------------
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load saved model files
# -----------------------------
vectorizer = joblib.load("vectorizer.pkl")
job_vectors = joblib.load("job_vectors.pkl")
jobs = joblib.load("jobs.pkl")


# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted + " "

    return text


# -----------------------------
# Home Page
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    score = None
    role = None
    found = []
    missing = []

    if request.method == "POST":

        # Get uploaded PDF
        pdf = request.files.get("resume")

        if pdf is None:
            return "No file uploaded."

        if pdf.filename == "":
            return "Please choose a PDF file."

        role = request.form["role"]

        # Safe filename
        filename = secure_filename(pdf.filename)

        # Full path
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save PDF
        pdf.save(filepath)

        print("Saved:", filepath)

        # Read PDF
        resume_text = extract_text(filepath)
        word_count = len(resume_text.split())
        char_count = len(resume_text)
        page_count = len(open(filepath, "rb").read().split(b"/Type /Page")) - 1

        # Convert resume into vector
        resume_vector = vectorizer.transform([resume_text])

        # Compare with all job roles
        similarities = cosine_similarity(
            resume_vector,
            job_vectors
        )

        # Selected role index
        index = jobs[jobs["Role"] == role].index[0]

        # Match score
        score = similarities[0][index] * 100

        # Skills
        job_skills = jobs.iloc[index]["Skills"].split()

        resume_lower = resume_text.lower()

        for skill in job_skills:

            if skill.lower() in resume_lower:
                found.append(skill)
            else:
                missing.append(skill)

    return render_template(
    "index.html",
    jobs=jobs["Role"],
    score=score,
    role=role,
    found=found,
    missing=missing,
    filename=filename if request.method == "POST" else "",
    word_count=word_count if request.method=="POST" else 0,
    char_count=char_count if request.method=="POST" else 0,
    page_count=page_count if request.method=="POST" else 0
)


# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)