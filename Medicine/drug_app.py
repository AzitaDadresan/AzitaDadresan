import streamlit as st

import numpy as np
import pandas as pd
import re
import os

from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import joblib

stop_words = [
    'a', 'about', 'after', 'all', 'also', 'an', 'and', 'any', 'as', 'at',
    'be', 'because', 'but', 'by', 'can', 'come', 'could', 'day', 'do', 'does', 'did', 'done',
    'dont', 'even', 'find', 'first', 'for', 'from', 'get', 'give', 'go', 'have', 'has', 'had',
    'he', 'her', 'here', 'him', 'his', 'how', 'i', 'ive', 'im', 'if', 'in', 'into',
    'it', 'its', 'just', 'know', 'like', 'look', 'make', 'man', 'many',
    'me', 'more', 'my', 'new', 'no', 'not', 'now', 'of', 'on', 'one',
    'only', 'or', 'other', 'our', 'out', 'people', 'say', 'see', 'she',
    'so', 'some', 'take', 'tell', 'than', 'that', 'the', 'their', 'them',
    'then', 'there', 'these', 'they', 'thing', 'think', 'this', 'those',
    'time', 'to', 'two', 'up', 'use', 'very', 'want', 'was', 'way', 'we', 'well',
    'what', 'when', 'which', 'who', 'will', 'with', 'would', 'year', 'you',
    'your'
]

# Load the CountVectorizer object
vectorizer = joblib.load(os.path.join('model', 'vectorizer.pkl'))

# Load the feature names 
feature_names = joblib.load(os.path.join('model', 'feature_names.pkl'))


# load the pytorch model
# Define the dimensions
input_dim = 75160
hidden_dim = 100
output_dim = 3264

# Define the neural network architecture
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        # x = self.softmax(x)
        return x

# Create the model instance
model = FeedForwardNN(input_dim, hidden_dim, output_dim)

# Load the saved model state dictionary
model.load_state_dict(torch.load(os.path.join('model', 'model.pt')))

# Set the model to evaluation mode
model.eval()


def make_prediction(model, test_sample, n=3):
    '''
    Return top-n predictions of the test_sample
    '''
    # model in evaluation mode
    model.eval()

    # convert test sample to torch tensor
    test_sample = test_sample.toarray()
    test_tensor = torch.FloatTensor(test_sample)
    
    # make prediction
    output = model(test_tensor)
    softmax = F.softmax(output, dim=1)[0]
    
    # Get the top n values and their indices
    top_values, top_indices = torch.topk(softmax, k=n)
    
    top_n_idx = top_indices[:n]
    
    result = []
    for idx in top_n_idx:
        result.append(feature_names[idx])

    return result


def main():
    
    # Title area
    st.markdown("<h1>Drug Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h2>A Machine Learning Approach for Disease Diagnosis</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Side panel with tabs
    st.sidebar.title("Options")
    tab = st.sidebar.radio("Select Option", ("Disease Diagnosis", "Test Accuracy"))


    if tab == "Disease Diagnosis":
        st.subheader("Enter Symptoms")
        symptoms = st.text_area("Please enter the symptoms of your disease", height=200)
        if st.button("Submit"):
            # business logic
            # keep only alphabets
            symptoms = re.sub(r'[^A-Za-z ]+', '', symptoms.lower())
            # remove stopwords
            filtered_tokens = [word for word in symptoms.split() if word not in stop_words]
            symptoms = ' '.join(filtered_tokens)
            # vectorize the symptoms
            test_data = vectorizer.transform([symptoms])

            # drug_recommendations 
            drug_recommendations = make_prediction(model, test_data)
            # drug_recommendations = ["Drug A", "Drug B", "Drug C"]  # Example output
            st.subheader("Possible Drugs")
            st.write(drug_recommendations)

    elif tab == "Test Accuracy":
        st.subheader("Test Accuracy")
        st.warning("The processing could take some time. Please wait.")
        csv_file = st.file_uploader("Upload a CSV file")
        if csv_file is not None:
            # load the file
            test_df = pd.read_csv(csv_file)
            # map drugs to symptoms:
            # Use two separate lists for that.
            test_symptoms = []
            test_drugs = []

            for idx, row in test_df.iterrows():
                # convert review text to lower case
                review = row['review'].lower().replace("&#039;", "'") \
                                                    .replace("&amp;", ' ') \
                                                    .replace("&quot;", ' ')
                
                # remove punctuations & digits
                review = re.sub(r'[^A-Za-z ]+', '', review)
                    
                # remove stopwords (common english words such as "the")
                filtered_tokens = [word for word in review.split() if word not in stop_words]
                review = ' '.join(filtered_tokens)
                
                # add the condition to the symptoms for better review
                condition = row['condition']
                if not pd.isna(condition) and "comment" not in condition:
                    # convert to lowercase
                    condition = condition.lower()
                    # keep only alphabets
                    condition = re.sub(r'[^A-Za-z ]+', '', condition)
                
                    # add to the review
                    review = condition + " " + review

                # process drug names
                drugName = row['drugName']
                # Replace "/" with "-" when surrounded by a number or a single alphabet
                drugName = re.sub(r'(\b\d|[A-Za-z])\s*\/\s*(\d|\b[A-Za-z])\b', r'\1-\2', drugName)
                
                # if multiple drugs in the same line separated by "/"
                # if multiple drugs in the same line separated by "/"
                for drug in drugName.split('/'):
                    test_drugs.append(drug.strip().lower())
                    test_symptoms.append(review)
            
            # vectorize the test data
            test_data = vectorizer.transform(test_symptoms)

            # check test accuracy
            correct = 0
            for sample, label in zip(test_data, test_drugs):
                # get the top-n outputs
                preds = make_prediction(model, sample, n=3)

                if label in preds:
                    correct += 1
        
            accuracy = np.round(100*correct/test_data.shape[0], 2)

            st.subheader("Computed Accuracy")
            st.text_area("Accuracy", value=str(accuracy) + "%")

if __name__ == "__main__":
    main()
