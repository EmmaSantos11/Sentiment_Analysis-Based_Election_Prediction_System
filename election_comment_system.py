import os
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import streamlit as st

# Initialize data structures
candidates = {}
comments_df = pd.DataFrame(columns=['candidate', 'comment', 'sentiment_score', 'sentiment'])
csv_file = 'comments_data.csv'

# Function to add candidates
def add_candidates(candidate1, candidate2):
    candidates[candidate1] = []
    candidates[candidate2] = []
    print(f"Candidates '{candidate1}' and '{candidate2}' added.")

# Function to add a new comment
def add_comment(candidate_name, comment):
    if candidate_name in candidates:
        candidates[candidate_name].append(comment)
        sentiment_score, sentiment = analyze_sentiment(comment)
        comments_df.loc[len(comments_df)] = [candidate_name, comment, sentiment_score, sentiment]
        save_comment_to_csv(candidate_name, comment, sentiment_score, sentiment)
        print(f"Comment added for candidate '{candidate_name}'.")
    else:
        print(f"Candidate '{candidate_name}' does not exist.")

# Function to analyze sentiment
def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0:
        sentiment = 'positive'
    elif sentiment_score == 0:
        sentiment = 'neutral'
    else:
        sentiment = 'negative'
    return sentiment_score, sentiment

# Function to plot word cloud
def plot_wordcloud(candidate_name):
    text = " ".join(comment for comment in candidates[candidate_name])
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {candidate_name}")
    plt.show()

# Function to plot sentiment distribution
def plot_sentiment_distribution():
    sentiment_counts = comments_df[comments_df['sentiment'] != 'neutral']['sentiment'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Sentiment Distribution (Positive and Negative only)')
    plt.show()

# Function to plot sentiment distribution for each candidate
def plot_candidate_sentiment_distribution(candidate_name):
    candidate_comments = comments_df[(comments_df['candidate'] == candidate_name) & 
                                     (comments_df['sentiment'] != 'neutral')]
    sentiment_counts = candidate_comments['sentiment'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title(f'Sentiment Distribution for {candidate_name} (Positive and Negative only)')
    plt.show()

# Function to get total number of comments
def total_comments():
    return len(comments_df)

# Function to save a single comment to CSV
def save_comment_to_csv(candidate_name, comment, sentiment_score, sentiment):
    with open(csv_file, 'a') as f:
        f.write(f"{candidate_name},{comment},{sentiment_score},{sentiment}\n")

# Function to save data to CSV
def save_data_to_csv():
    comments_df.to_csv(csv_file, index=False)
    print("Data saved to comments_data.csv")

# Function to load comments from CSV
def load_comments_from_csv():
    global comments_df
    if os.path.exists(csv_file):
        comments_df = pd.read_csv(csv_file)
        print("Comments loaded from comments_data.csv")
    else:
        print("No existing comments data found.")

# Function to handle file upload and processing
def handle_file_upload(file):
    uploaded_df = pd.read_csv(file)
    candidate_name = os.path.splitext(os.path.basename(file.name))[0]  # Use file name as candidate name
    add_candidates(candidate_name, "")  # Add candidate
    for _, row in uploaded_df.iterrows():
        comment = row['comment']
        add_comment(candidate_name, comment)

# Main function to run the election comment system
def election_comment_system():
    st.title("Election Sentiment Analysis")
    
    load_comments_from_csv()

    st.header("Upload CSV File(s) with Comments")
    uploaded_files = st.file_uploader("Choose CSV file(s)", type="csv", accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            handle_file_upload(file)
        st.success("Files uploaded and processed successfully")

    st.header("Current Candidates")
    st.write(list(candidates.keys()))

    st.header("Enter Comments for Candidates")
    candidate_name = st.selectbox("Select Candidate", list(candidates.keys()))
    comment = st.text_area("Enter Comment")

    if st.button("Submit Comment"):
        add_comment(candidate_name, comment)

    if st.button("End Comment Collection"):
        save_data_to_csv()

    if st.button("Run Sentiment Analysis"):
        st.header("Sentiment Analysis Results")
        if not comments_df.empty:
            st.subheader("Sentiment Distribution")
            plot_sentiment_distribution()
            
            for candidate_name in candidates.keys():
                st.subheader(f"Sentiment Distribution for {candidate_name}")
                plot_candidate_sentiment_distribution(candidate_name)
                st.subheader(f"Word Cloud for {candidate_name}")
                plot_wordcloud(candidate_name)
        else:
            st.warning("No comments available to analyze.")

if __name__ == "__main__":
    election_comment_system()