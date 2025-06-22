import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

nltk.download("stopwords")
nltk.download("wordnet")
nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Common words dictionary for typo correction
COMMON_WORDS = {
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'among', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
    'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'am',
    'is', 'are', 'was', 'were', 'being', 'been', 'be', 'have', 'has', 'had', 'do',
    'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
    'news', 'article', 'report', 'said', 'says', 'told', 'reuters', 'company', 'market',
    'business', 'economic', 'financial', 'politics', 'government', 'political', 'sports',
    'technology', 'health', 'medical', 'science', 'research', 'study', 'data', 'system'
}

def levenshtein_distance(s1, s2):

    #Calculate the Levenshtein distance between two strings.

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def find_closest_word(word, word_set, threshold=2):

    #Find the closest word in the word set using Levenshtein distance.

    if word in word_set:
        return word
    
    min_distance = float('inf')
    closest_word = word
    
    for correct_word in word_set:
        # Only check words with similar length (optimization)
        if abs(len(word) - len(correct_word)) <= threshold:
            distance = levenshtein_distance(word, correct_word)
            if distance <= threshold and distance < min_distance:
                min_distance = distance
                closest_word = correct_word
    
    return closest_word

def correct_typos_advanced(text):
 
    #Advanced typo correction using Levenshtein distance.

    words = text.split()
    corrected_words = []
    
    for word in words:
        # Skip very short words and numbers
        if len(word) <= 2 or word.isdigit():
            corrected_words.append(word)
            continue
        
        # Find closest match in common words
        corrected_word = find_closest_word(word.lower(), COMMON_WORDS, threshold=2)
        corrected_words.append(corrected_word)
    
    return " ".join(corrected_words)

def correct_typos_simple(text):

    #Simple dictionary-based typo correction (original method).
    
    corrections = {
        "reuter": "reuters", 
        "teh": "the", 
        "recieve": "receive",
        "occured": "occurred",
        "seperate": "separate",
        "definately": "definitely",
        "accomodate": "accommodate",
        "begining": "beginning",
        "beleive": "believe",
        "publically": "publicly"
    }
    words = text.split()
    corrected = [corrections.get(w.lower(), w) for w in words]
    return " ".join(corrected)

def preprocess(text, use_advanced_correction=True):

    #Comprehensive text preprocessing pipeline.
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # Apply typo correction
    if use_advanced_correction:
        text = correct_typos_advanced(text)
    else:
        text = correct_typos_simple(text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    
    # Apply stemming
    tokens = [stemmer.stem(w) for w in tokens]
    
    # Apply lemmatization
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return " ".join(tokens)



