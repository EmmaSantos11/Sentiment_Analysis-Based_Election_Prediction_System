# download_nltk_resources.py

import nltk

def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

if __name__ == "__main__":
    download_nltk_resources()
