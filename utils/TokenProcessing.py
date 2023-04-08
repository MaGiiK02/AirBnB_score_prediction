"""__summary
A collection of methods to work with tokens lists
"""
import itertools
import string
import re
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Snipped to execute the download of nltk moule only if not present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download(['wordnet', 'omw-1.4'])

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

# Some token are related to the domain this handcrafted function deal with removing them
# DOMAIN_TOKENS =  ['@user']
# def remove_domain_tokens(tokens):
#   return list(filter(lambda t: t not in DOMAIN_TOKENS, tokens))

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
