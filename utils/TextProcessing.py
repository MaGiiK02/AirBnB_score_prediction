"""__summary
A collection of methods to work with pure text
"""
import itertools
import re
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk

import re
from nltk.corpus import stopwords
import string

# define the stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))


# Define regular expression pattern to match stop words
stop_word_regex = r'\b(?:{})\b'.format('|'.join(stop_words))

#  Defining regex for emojis
smileemoji = r"[8:=;]['`\-]?[)d]+"
sademoji = r"[8:=;]['`\-]?\(+"
neutralemoji = r"[8:=;]['`\-]?[\/|l*]"
lolemoji = r"[8:=;]['`\-]?p+"

# pattern for unicode characters
pattern_unicode = r'\\u[0-9A-Fa-f]{4}'

lemmatizer = nltk.stem.WordNetLemmatizer()
ps = nltk.stem.PorterStemmer()


stop_words = set(stopwords.words('english'))

def uncontract(text):    
  text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
  text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
  text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
  text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)
  
  text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
  text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
  text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
  text = re.sub(r"(\b)([Tt]here)'s", r"\1\2 is", text)
  text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
  text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
  text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
  
  return text

def lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def tokenize(sentence):
  return word_tokenize(sentence)


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
