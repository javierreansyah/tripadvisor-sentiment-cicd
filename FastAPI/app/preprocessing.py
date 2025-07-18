import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# Download required NLTK data
def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded."""
    downloads = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),  
        ('corpora/wordnet', 'wordnet')
    ]
    
    for path, name in downloads:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK data: {name}")
            nltk.download(name, quiet=True)

# Initialize NLTK data
ensure_nltk_data()

def preprocess_for_vectorizer(text):
    """
    Preprocessing function for the TfidfVectorizer.
    This function will be used inside the pipeline's vectorizer.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Standardize text
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"http", "", text)
    text = re.sub(r"@/S+", "", text)
    text = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", text)
    text = text.replace("@", " at ")
    text = text.lower()
    
    # Tokenize, remove stopwords, and lemmatize
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)