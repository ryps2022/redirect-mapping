
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import base64

# Define a function to perform matching and generate similarity scores
def perform_matching(origin_df, destination_df, selected_columns):
    # Combine the selected columns into a single text column for vectorization
    origin_df['combined_text'] = origin_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)
    destination_df['combined_text'] = destination_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)

    # Use a pre-trained model for embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Vectorize the combined text
    origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
    destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

    # Create a FAISS index
    dimension = origin_embeddings.shape[1]  # The dimension of vectors
    faiss_index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
    faiss_index.add(destination_embeddings.astype('float32'))  # Add destination vectors to the index

    # Perform the search for the nearest neighbors
    D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)  # k=1 finds the closest match

    # Calculate similarity score (1 - normalized distance)
    similarity_scores = 1 - (D / np.max(D))

    # Create the output DataFrame with similarity score instead of distance
    matches_df = pd.DataFrame({
        'origin_url': origin_df['Address'],
        'matched_url': destination_df['Address'].iloc[I.flatten()].values,
        'similarity_score': np.round(similarity_scores.flatten(), 4)  # Rounded for better readability
    })

    return matches_df

# Main function to run the Streamlit app
def main():
    st.title("Similarity Matcher App")

    # Upload the origin.csv and destination.csv files
    st.write("Please upload the origin.csv file.")
    origin_df = st.file_uploader("Upload Origin CSV", type=['csv'])
    st.write("Please upload the destination.csv file.")
    destination_df = st.file_uploader("Upload Destination CSV", type=['csv'])

    # Select columns for similarity matching
    if origin_df is not None and destination_df is not None:
        origin_df = pd.read_csv(origin_df)
        destination_df = pd.read_csv(destination_df)

        common_columns = list(set(origin_df.columns) & set(destination_df.columns))
        selected_columns = st.multiselect("Select the columns you want to include for similarity matching:", common_columns)

        if st.button("Let's Go!"):
            if not selected_columns:
                st.warning("Please select at least one column to continue.")
            else:
                matches_df = perform_matching(origin_df, destination_df, selected_columns)
                st.write("Matching complete.")

                # Download the results as a CSV file
                csv = matches_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding for data
                href = f'<a href="data:file/csv;base64,{b64}" download="matching_results.csv">Download Matching Results CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
