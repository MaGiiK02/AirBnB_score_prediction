"""__summary
A collection of methods to work with list of documents Tokens
"""

import itertools
import re
from collections import Counter

# Remove tokens that appear less than the lesser_eq_than between all the documetns
def remove_rare_tokens(docs, lesser_eq_than):
    tokens_linear = itertools.chain.from_iterable(docs)
    token_freq = dict(Counter(tokens_linear))
    tokens_appeared_leq = set([t for t, v in token_freq.items() if v <= lesser_eq_than])
    return [list(filter(lambda t: t not in tokens_appeared_leq, doc)) for doc in docs]

# Remove dockuments that have no tokens left
def remove_empty(docs):
    return list(filter(lambda d: len(d)>0, docs))