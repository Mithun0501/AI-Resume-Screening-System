import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

# Load job dataset
jobs = pd.read_csv("jobs.csv")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

job_vectors = vectorizer.fit_transform(jobs["Skills"])

# Save
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(job_vectors, "job_vectors.pkl")
joblib.dump(jobs, "jobs.pkl")

print("Resume Screening Model Created Successfully!")