"""__summary
A collection of methods to work with tokens lists
"""
import itertools
import string
import re
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# Snipped to execute the download of nltk moule only if not present
nltk.download("stopwords")
nltk.download('punkt')

stop_words = set(stopwords.words('english'))


# Remplace urls and links tokens with the _URL placeholder
def tokenise_URLS(tokens):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_regex = re.compile(url_regex)
    return [re.sub(url_regex, '_URL', t ) for t in tokens]

# Remplace E-Mail tokens with the _EMAIL placeholder
def tokenise_emails(tokens):
  email_regex = '^[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*@[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*$'
  email_regex = re.compile(email_regex)
  return [re.sub(email_regex, '_EMAIL', t ) for t in tokens]


# Remove all the punctuation only tokens
def remove_puctuation_only(tokens):
  return list(filter(lambda t: not all(c in string.punctuation for c in t), tokens))

# Remove Numbers
def remove_numbers(tokens):
  return list(filter(lambda t: not t.isnumeric() , tokens))

# remove the stop words from the tokens
def remove_stop_words(tokens):
  stop_word_regex = '|'.join(['^{}$'.format(s) for s in stopwords.words('english')])
  stop_word_regex = re.compile(stop_word_regex)
  return list(filter(lambda t: not stop_word_regex.match(t), tokens))

# Lemmatixe the tokens using WordNetLemmatizer
def lemmatize(tokens):
  lemmatizer = WordNetLemmatizer()
  return [lemmatizer.lemmatize(t) for t in tokens]

# Stemmatize the token using PorterStemmer
def stemmatize(tokens):
  ps = PorterStemmer()
  return [ps.stem(t) for t in tokens]

# Remove empty tokens
def filter_empty(tokens):
  return list(filter(lambda t: t is not None and t != "", tokens))


# we create a function to pre process the text and we have a default value for normalization. we can change it to stemming or lemmatization
# to check which one is better for our model

import string
import re
from typing import List
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Compile regex patterns once
pattern_unicode = re.compile(r'[^\x00-\x7F]+', re.UNICODE)
stop_word_regex = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
smileemoji = re.compile(u'[\U0001F600-\U0001F64F]')
sademoji = re.compile(u'[\U0001F300-\U0001F5FF]')
neutralemoji = re.compile(u'[\U0001F610-\U0001F61F]')
lolemoji = re.compile(u'[\U0001F600-\U0001F600]')

# Initialize lemmatizer and stemmer outside the function
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def preprocess_text(text: List[str], normalization: str = "stemming") -> List[str]:
    # Convert all text to lower case
    text = [t.lower() for t in text]

    # Replace all emojis
    text = [re.sub(r'<3', '<heart>', t) for t in text]
    text = [re.sub(smileemoji, '<smile>', t) for t in text]
    text = [re.sub(sademoji, '<sadface>', t) for t in text]
    text = [re.sub(neutralemoji, '<neutralface>', t) for t in text]
    text = [re.sub(lolemoji, '<lolface>', t) for t in text]

    # Remove unicode characters
    text = [re.sub(pattern_unicode, '', t) for t in text]

    # Remove consecutive characters
    text = [re.sub(r'(.)\1+', r'\1\1', t) for t in text]

    # Remove short words
    clean_tokens = [w for w in text if len(w) >= 3]

    # Apply stemming or lemmatization
    if normalization == "stemming":
        clean_tokens = [ps.stem(t) for t in clean_tokens]
    else:
        clean_tokens = [lemmatizer.lemmatize(t) for t in clean_tokens]

    return clean_tokens
