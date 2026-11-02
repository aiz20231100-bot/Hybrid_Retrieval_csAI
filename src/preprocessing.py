import re
from functools import lru_cache

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()
TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = set("""
a an and are as at be been but by for from has have had he her his i if in into is it its
of on or our she so such than that the their them then there these they this those to was
we were what when where which while who will with would you your can could may might also
not no do does did done using used use based via per et al fig figure table
""".split())


def clean_text(text):
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=200_000)
def _stem(token):
    return _stemmer.stem(token)


def tokenize(text, stem=True):
    tokens = TOKEN_RE.findall(text.lower())
    tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
    return [_stem(t) for t in tokens] if stem else tokens