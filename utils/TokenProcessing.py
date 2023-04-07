"""__summary
A collection of methods to work with tokens lists
"""
import itertools
import re
from collections import Counter

def tokenise_URLS(tokens):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_regex = re.compile(url_regex)
    return [re.sub(url_regex, '_URL', t ) for t in tokens]

def tokenise_emails(tokens):
  email_regex = '^[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*@[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*$'
  email_regex = re.compile(email_regex)
  return [re.sub(email_regex, '_EMAIL', t ) for t in tokens]