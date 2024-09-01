import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load the dataset
df = pd.read_csv('mercortask.csv')

# Define a function to calculate similarity scores
def calculate_similarities(text1, text2):
    # Combine texts
    combined_text = f"{text1} {text2}"
    
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([combined_text])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    return similarity[0][0]

# Function to extract features using a pre-trained language model
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()[0]
    return embeddings

# Prepare data
X = []
y = []

for _, row in df.iterrows():
    # Extract features
    role_embedding = get_embeddings(row['role'], tokenizer, model)
    candidate_a_transcript_embedding = get_embeddings(row['candidateATranscript'], tokenizer, model)
    candidate_b_transcript_embedding = get_embeddings(row['candidateBTranscript'], tokenizer, model)
    
    # Calculate similarities
    sim_role_a = calculate_similarities(row['role'], row['candidateATranscript'])
    sim_role_b = calculate_similarities(row['role'], row['candidateBTranscript'])
    
    # Combine features
    features = np.concatenate([
        role_embedding,
        candidate_a_transcript_embedding,
        candidate_b_transcript_embedding,
        [sim_role_a, sim_role_b]
    ])
    
    X.append(features)
    y.append(1 if row['winnerId'] == row['candidateAId'] else 0)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Evaluate the model
accuracy = rfc.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Use the best model to make predictions on test set
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(X_test)

# Calculate final accuracy
final_accuracy = np.mean(test_predictions == y_test)
print("Final Test Accuracy:", final_accuracy)

# Save the best model
import pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Load the saved model
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Make predictions on new data
new_data = pd.DataFrame({
    'candidateAId': ['123'],
    'candidateBId': ['456'],
    'candidateATranscript': ['This is candidate A\'s transcript'],
    'candidateBTranscript': ['This is candidate B\'s transcript'],
    'role': ['Software Engineer']
})

new_features = []
for _, row in new_data.iterrows():
    role_embedding = get_embeddings(row['role'], tokenizer, model)
    candidate_a_transcript_embedding = get_embeddings(row['candidateATranscript'], tokenizer, model)
    candidate_b_transcript_embedding = get_embeddings(row['candidateBTranscript'], tokenizer, model)
    
    sim_role_a = calculate_similarities(row['role'], row['candidateATranscript'])
    sim_role_b = calculate_similarities(row['role'], row['candidateBTranscript'])
    
    features = np.concatenate([
        role_embedding,
        candidate_a_transcript_embedding,
        candidate_b_transcript_embedding,
        [sim_role_a, sim_role_b]
    ])
    
    new_features.append(features)

new_predictions = loaded_model.predict(new_features)
print("Prediction:", "Candidate A" if new_predictions[0] == 1 else "Candidate B")

