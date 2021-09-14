# coding=utf-8

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

# Default list of stop words (words so common it's usually wasteful to index
# them). This list is used by the StopFilter class, which allows you to supply
# an optional list to override this one.
from itertools import chain
from typing import Generator, FrozenSet, Dict

from whooshy.core import Composable, Token

STOP_WORDS = frozenset(('a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can',
                        'for', 'from', 'have', 'if', 'in', 'is', 'it', 'may',
                        'not', 'of', 'on', 'or', 'tbd', 'that', 'the', 'this',
                        'to', 'us', 'we', 'when', 'will', 'with', 'yet',
                        'you', 'your'))


# todo: url_pattern to filtering urls maybe useful (matt already does it)


class Filter(Composable):
    """Base class for Filter objects. A Filter subclass must implement a
    filter() method that takes a single argument, which is an iterator of Token
    objects, and yield a series of Token objects in return.

    Filters that do morphological transformation of tokens (e.g. stemming)
    should set their ``is_morph`` attribute to True.
    """

    def __eq__(self, other: "Filter") -> bool:
        return (other
                and self.__class__ is other.__class__
                and self.__dict__ == other.__dict__)

    def __ne__(self, other: "Filter") -> bool:
        return not self == other

    def __call__(self, tokens: Generator[Token, None, None]):
        raise NotImplementedError


class PassFilter(Filter):
    """An identity filter: passes the tokens through untouched.
    """

    def __call__(self, tokens: Generator[Token, None, None]):
        return tokens


class LoggingFilter(Filter):
    """Prints the contents of every filter that passes through as a debug
    log entry.
    """

    def __init__(self, logger: "Logger" = None):
        """
        :param target: the logger to use. If omitted, the "whooshy.analysis"
            logger is used.
        """

        if logger is None:
            import logging
            logger = logging.getLogger("whooshy.analysis")
        self.logger = logger

    def __call__(self, tokens: Generator[Token, None, None]):
        logger = self.logger
        for t in tokens:
            logger.debug(repr(t))
            yield t


class MultiFilter(Filter):
    """Chooses one of two or more sub-filters based on the 'mode' attribute
    of the token stream.
    """

    default_filter = PassFilter()

    def __init__(self, **kwargs):
        """Use keyword arguments to associate mode attribute values with
        instantiated filters.

        This class expects that the value of the mode attribute is consistent
        among all tokens in a token stream.
        """
        self.filters = kwargs

    def __eq__(self, other: "MultiFilter"):
        return (other
                and self.__class__ is other.__class__
                and self.filters == other.filters)

    def __call__(self, tokens: Generator[Token, None, None]):
        # Only selects on the first token
        t = next(tokens)
        filter = self.filters.get(t.mode, self.default_filter)
        # noinspection PyTypeChecker
        return filter(chain([t], tokens))


class LowercaseFilter(Filter):
    """Uses unicode.lower() to lowercase token text.

    >>> rext = RegexTokenizer()
    >>> stream = rext("This is a TEST")
    >>> [token.text for token in LowercaseFilter(stream)]
    ["this", "is", "a", "test"]
    """

    def __call__(self, tokens: Generator[Token, None, None]):
        for t in tokens:
            t.text = t.text.lower()
            yield t


class StripFilter(Filter):
    """Calls unicode.strip() on the token text.
    """

    def __call__(self, tokens: Generator[Token, None, None]):
        for t in tokens:
            t.text = t.text.strip()
            yield t


class StopFilter(Filter):
    """Marks "stop" words (words too common to index) in the stream (and by
    default removes them).

    Make sure you precede this filter with a :class:`LowercaseFilter`.

    >>> stopper = RegexTokenizer() | StopFilter()
    >>> [token.text for token in stopper(u"this is a test")]
    ["test"]
    >>> es_stopper = RegexTokenizer() | StopFilter(lang="spanish")
    >>> [token.text for token in es_stopper(u"el lapiz es en la mesa")]
    ["lapiz", "mesa"]

    You can use :func:`whooshy.util.has_stopwords` to check if a given language
    has a stop word list available.
    """

    def __init__(self, stop_list: FrozenSet[str] = STOP_WORDS, minsize: int = 2, maxsize: int = None,
                 renumber: bool = False, lang: str = None):
        """
        :param stop_list: A collection of words to remove from the stream.
            This is converted to a frozenset. The default is a list of
            common English stop words.
        :param minsize: The minimum length of token texts. Tokens with
            text smaller than this will be stopped. The default is 2.
        :param maxsize: The maximum length of token texts. Tokens with text
            larger than this will be stopped. Use None to allow any length.
        :param renumber: Change the 'pos' attribute of unstopped tokens
            to reflect their position with the stopped words removed.
        :param lang: Automatically get a list of stop words for the given
            language {'hungarian', 'swedish', 'kazakh', 'norwegian', 'finnish', 'arabic', 'indonesian', 'portuguese',
            'turkish', 'azerbaijani', 'slovene', 'spanish', 'danish', 'nepali', 'romanian', 'greek', 'dutch', 'tajik',
            'german', 'english', 'russian', 'french', 'italian'}
        """

        stops = set()
        if stop_list:
            stops.update(stop_list)
        if lang:
            from nltk.corpus import stopwords
            try:
                stops.update(stopwords.words(lang))
            except LookupError:  # just ignores if not added
                pass

        self.stops = frozenset(stops)
        self.min = minsize
        self.max = maxsize
        self.renumber = renumber

    def __eq__(self, other: "StopFilter"):
        return (other
                and self.__class__ is other.__class__
                and self.stops == other.stops
                and self.min == other.min
                and self.renumber == other.renumber)

    def __call__(self, tokens: Generator[Token, None, None]):
        stop_list = self.stops
        minsize = self.min
        maxsize = self.max
        renumber = self.renumber

        pos = None
        for t in tokens:  # for each token analysed
            text = t.text
            if (len(text) >= minsize
                    and (maxsize is None or len(text) <= maxsize)
                    and text not in stop_list):
                # This is not a stop word
                if renumber and t.positions:
                    if pos is None:
                        pos = t.pos
                    else:
                        pos += 1
                        t.pos = pos
                t.stopped = False
                yield t
            else:
                # This is a stop word
                if not t.remove_stopwords:
                    # This IS a stop word, but we're not removing them
                    t.stopped = True
                    yield t


class CharsetFilter(Filter):
    """Translates the text of tokens by calling unicode.translate() using the
    supplied character mapping object. This is useful for case and accent
    folding.

    The ``whooshy.support.charset`` module has a useful map for accent folding.

    >>> from whooshy.support.charset import accent_map
    >>> retokenizer = RegexTokenizer()
    >>> chfilter = CharsetFilter(accent_map)
    >>> [t.text for t in chfilter(retokenizer(u'café'))]
    [u'cafe']

    Another way to get a character mapping object is to convert a Sphinx
    charset table file using
    :func:`whooshy.support.charset.charset_table_to_dict`.

    >>> from whooshy.support.charset import charset_table_to_dict
    >>> from whooshy.support.charset import default_charset
    >>> retokenizer = RegexTokenizer()
    >>> charmap = charset_table_to_dict(default_charset)
    >>> chfilter = CharsetFilter(charmap)
    >>> [t.text for t in chfilter(retokenizer(u'Stra\\xdfe'))]
    [u'strase']

    The Sphinx charset table format is described at
    http://www.sphinxsearch.com/docs/current.html#conf-charset-table.
    """

    __inittypes__ = dict(char_map=dict)

    def __init__(self, char_map: Dict[int, chr]):
        """
        :param char_map: a dictionary mapping from integer character numbers to
            unicode characters, as required by the unicode.translate() method.
        """

        self.char_map = char_map

    def __eq__(self, other: "CharsetFilter"):
        return (other
                and self.__class__ is other.__class__
                and self.char_map == other.char_map)

    def __call__(self, tokens: Generator[Token, None, None]):
        assert hasattr(tokens, "__iter__")
        char_map = self.char_map
        for t in tokens:
            # noinspection PyTypeChecker
            t.text = t.text.translate(char_map)  # translate using the prebuilt table
            yield t
