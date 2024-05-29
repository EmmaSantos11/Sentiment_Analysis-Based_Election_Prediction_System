import nltk
import ssl
import os
from urllib.request import build_opener, HTTPSHandler

# Path to the cacert.pem file
cert_file_path = os.path.join(os.path.dirname(__file__), 'cacert.pem')

# Set up SSL context
ssl_context = ssl.create_default_context(cafile=cert_file_path)

# Create an opener with the SSL context
opener = build_opener(HTTPSHandler(context=ssl_context))

# Install the opener
import urllib.request
urllib.request.install_opener(opener)

# Download NLTK data
nltk.download('vader_lexicon')
