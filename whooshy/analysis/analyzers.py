# Copyright 2007 Matt Chaput. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY MATT CHAPUT ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL MATT CHAPUT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Matt Chaput.

from whooshy.core import Composable, CompositionError
from whooshy.analysis.tokenizers import Tokenizer
from whooshy.analysis.filters import LowercaseFilter
from whooshy.analysis.filters import StopFilter, STOP_WORDS
from whooshy.analysis.morph import StemFilter
from whooshy.analysis.tokenizers import default_pattern
from whooshy.analysis.tokenizers import CommaSeparatedTokenizer
from whooshy.analysis.tokenizers import IDTokenizer
from whooshy.analysis.tokenizers import RegexTokenizer
from whooshy.analysis.tokenizers import SpaceSeparatedTokenizer


# Analyzers

class Analyzer(Composable):
    """ Abstract base class for analyzers.
    """

    def __repr__(self):
        return "%s()" % self.__class__.__name__

    def __eq__(self, other):
        return (other
                and self.__class__ is other.__class__
                and self.__dict__ == other.__dict__)

    def __call__(self, value, **kwargs):
        raise NotImplementedError

    def clean(self):
        pass


class CompositeAnalyzer(Analyzer):
    """
    Chain analysers.
    """

    def __init__(self, *components):
        self.items = []

        for comp in components:
            if isinstance(comp, CompositeAnalyzer):
                self.items.extend(comp.items)
            else:
                self.items.append(comp)

        # Tokenizers must start a chain, and then only filters after that
        # (because analyzers take a string and return a generator of tokens,
        # and filters take and return generators of tokens)
        for item in self.items[1:]:
            if isinstance(item, Tokenizer):
                raise CompositionError("Only one tokenizer allowed at the start"
                                       " of the analyzer: %r" % self.items)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join(repr(item) for item in self.items))

    def __call__(self, value, no_morph=False, **kwargs):
        items = self.items
        # Start with tokenizer
        gen = items[0](value, **kwargs)
        # Run filters
        for item in items[1:]:
            if not (no_morph and hasattr(item, "is_morph") and item.is_morph):
                gen = item(gen)
        return gen

    def __getitem__(self, item):
        return self.items.__getitem__(item)

    def __len__(self):
        return len(self.items)

    def __eq__(self, other):
        return (other
                and self.__class__ is other.__class__
                and self.items == other.items)

    def clean(self):
        for item in self.items:
            if hasattr(item, "clean"):
                item.clean()

    def has_morph(self):
        return any(item.is_morph for item in self.items)


def KeywordAnalyzer(lowercase: bool = False, commas: bool = False):
    """Parses whitespace- or comma-separated tokens.

    >>> ana = KeywordAnalyzer()
    >>> [token.text for token in ana("Hello there, this is a TEST")]
    ["Hello", "there,", "this", "is", "a", "TEST"]

    :param lowercase: whether to lowercase the tokens.
    :param commas: if True, items are separated by commas rather than
        whitespace.
    """

    if commas:
        tokenizer = CommaSeparatedTokenizer()
    else:
        tokenizer = SpaceSeparatedTokenizer()
    if lowercase:
        tokenizer = tokenizer | LowercaseFilter()
    return tokenizer


def SimpleAnalyzer(expression=default_pattern, gaps=False):
    """Composes a RegexTokenizer with a LowercaseFilter.

    >>> ana = SimpleAnalyzer()
    >>> [token.text for token in ana("Hello there, this is a TEST")]
    ["hello", "there", "this", "is", "a", "test"]

    :param expression: The regular expression pattern to use to extract tokens.
    :param gaps: If True, the tokenizer *splits* on the expression, rather
        than matching on the expression.
    """

    return RegexTokenizer(expression=expression, gaps=gaps) | LowercaseFilter()


def StandardAnalyzer(expression=default_pattern, stop_words=STOP_WORDS,
                     minsize=2, maxsize=None, gaps=False):
    """Composes a RegexTokenizer with a LowercaseFilter and optional
    StopFilter.

    >>> ana = StandardAnalyzer()
    >>> [token.text for token in ana("Testing is testing and testing")]
    ["testing", "testing", "testing"]

    :param expression: The regular expression pattern to use to extract tokens.
    :param stop_words: A list of stop words. Set this to None to disable
        the stop word filter.
    :param minsize: Words smaller than this are removed from the stream.
    :param maxsize: Words longer that this are removed from the stream.
    :param gaps: If True, the tokenizer *splits* on the expression, rather
        than matching on the expression.
    """

    ret = RegexTokenizer(expression=expression, gaps=gaps)
    chain = ret | LowercaseFilter()
    if stop_words is not None:
        chain = chain | StopFilter(stop_list=stop_words, minsize=minsize,
                                   maxsize=maxsize)
    return chain


def StemmingAnalyzer(expression=default_pattern, stoplist=STOP_WORDS,
                     minsize=2, maxsize=None, gaps=False, stemmer=None,
                     ignore=None):
    """Composes a RegexTokenizer with a lower case filter, an optional stop
    filter, and a stemming filter.

    >>> ana = StemmingAnalyzer()
    >>> [token.text for token in ana("Testing is testing and testing")]
    ["test", "test", "test"]

    :param expression: The regular expression pattern to use to extract tokens.
    :param stoplist: A list of stop words. Set this to None to disable
        the stop word filter.
    :param minsize: Words smaller than this are removed from the stream.
    :param maxsize: Words longer that this are removed from the stream.
    :param gaps: If True, the tokenizer *splits* on the expression, rather
        than matching on the expression.
    :param stemmer: NLTK stemming algorithm
    :param ignore: a set of words to not stem.
    """

    ret = RegexTokenizer(expression=expression, gaps=gaps)
    chain = ret | LowercaseFilter()
    if stoplist is not None:
        chain = chain | StopFilter(stop_list=stoplist, minsize=minsize,
                                   maxsize=maxsize)
    return chain | StemFilter(stemmer=stemmer, ignore=ignore)


def LanguageAnalyzer(lang, expression=default_pattern, gaps=False,
                     ignore=None):
    """Configures a simple analyzer for the given language, with a
    LowercaseFilter, StopFilter, and StemFilter.

    >>> ana = LanguageAnalyzer("es")
    >>> [token.text for token in ana("Por el mar corren las liebres")]
    ['mar', 'corr', 'liebr']

    The list of available languages is in `whoosh.lang.languages`.
    You can use :func:`whoosh.lang.has_stemmer` and
    :func:`whoosh.lang.has_stopwords` to check if a given language has a
    stemming function and/or stop word list available.

    :param expression: The regular expression pattern to use to extract tokens.
    :param gaps: If True, the tokenizer *splits* on the expression, rather
        than matching on the expression.
    :param ignore: a set of words to not stem.
    """
    # Make the start of the chain
    chain = (RegexTokenizer(expression=expression, gaps=gaps)
             | LowercaseFilter())

    # Add a stop word filter and a stemming filter
    # noinspection PyBroadException
    try:
        chain = chain | StopFilter(lang=lang)
        chain = chain | StemFilter(lang=lang)
    except Exception:
        pass

    return chain
