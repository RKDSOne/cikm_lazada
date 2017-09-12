'''
Pre-process corpus
'''


import re
import unicodedata
import codecs

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def process_data(text):

    text = text.lower()

    # remove numbers
    text = re.sub('[0-9]', '', text)

    # remove stop words
    stop_words = stopwords.words('english')
    words = text.split()
    words = [w for w in words if w not in stop_words]

    # wnl = WordNetLemmatizer()
    # lemmatize word if the length is greater than 4, if not do nothing to the word
    # words = [wnl.lemmatize(w) if len(w) > 4 else w for w in words]

    # combine words back into a single string
    return ' '.join(words)
