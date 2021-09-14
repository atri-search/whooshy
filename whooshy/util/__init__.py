from whooshy.util.text import *


def has_stopwords(lang: str) -> bool:
    """
    Verifies if the language has a default stopwords list
    :param lang:  You can use :func:`whooshy.util.has_stemmer` to check if a given language has
    a stemming function available
    """
    from nltk.corpus import stopwords
    return lang in stopwords.fileids()


def get_stemmer_by_lang(lang: str):
    """
    Get the recommended stemmer by language
    :param lang: You can use :func:`whooshy.util.has_stemmer` to check if a given language has
    a stemming function available
    """
    import nltk.stem as stem
    if not lang:
        return stem.porter.PorterStemmer()

    if lang == 'portuguese':
        return stem.rslp.RSLPStemmer()
    return stem.porter.PorterStemmer()
