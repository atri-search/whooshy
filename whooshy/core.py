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

from typing import Iterable, AnyStr


# Exceptions


class CompositionError(Exception):
    pass


# Utility functions

def unstopped(token_stream: Iterable["Token"]):
    """Removes tokens from a token stream where token.stopped = True.
    """
    return (t for t in token_stream if not t.stopped)


def tokenize(text_stream: Iterable[bytes], positions: bool = False, chars: bool = False, start_pos: int = 0,
             start_char: int = 0, **kwargs):
    """Takes a sequence of unicode strings and yields a series of Token objects
    (actually the same Token object over and over, for performance reasons),
    with the attributes filled in with reasonable values (for example, if
    ``positions`` or ``chars`` is True, the function assumes each token was
    separated by one space).
    """

    pos = start_pos
    char = start_char
    t = Token(positions=positions, chars=chars, **kwargs)

    for text in text_stream:
        t.text = text

        if positions:
            t.pos = pos
            pos += 1

        if chars:
            t.start_char = char
            char = char + len(text)
            t.end_char = char

        yield t


class Token(object):
    """
    Represents a "token" (usually a word) extracted from the source text being
    indexed.

    See "Advanced analysis" in the user guide for more information.

    Because object instantiation in Python is slow, tokenizers should create
    ONE SINGLE Token object and YIELD IT OVER AND OVER, changing the attributes
    each time.

    This trick means that consumers of tokens (i.e. filters) must never try to
    hold onto the token object between loop iterations, or convert the token
    generator into a list. Instead, save the attributes between iterations,
    not the object::

        def RemoveDuplicatesFilter(self, stream):
            # Removes duplicate words.
            last_text = None
            for token in stream:
                # Only yield the token if its text doesn't
                # match the previous token.
                if last_text != token.text:
                    yield token
                last_text = token.text

    ...or, call token.copy() to get a copy of the token object.
    """

    def __init__(self, positions: bool = False, chars: bool = False, remove_stopwords: bool = True, mode: str = '',
                 **kwargs):
        """
        :param positions: Whether tokens should have the token position in the
            'pos' attribute.
        :param chars: Whether tokens should have character offsets in the
            'start_char' and 'end_char' attributes.
        :param remove_stopwords: whether to remove stop words from the stream (if
            the tokens pass through a stop filter).
        :param mode: contains a string describing the purpose for which the
            analyzer is being called, i.e. 'index' or 'query'.
        """

        self._positions: bool = positions
        self._chars: bool = chars
        self._remove_stopwords: bool = remove_stopwords
        self._mode: str = mode
        self.boost: float = 1.0
        self.stopped: bool = False
        self.text: AnyStr = ''
        self.original_text: AnyStr = ''
        self.pos: int = 0
        self.start_char: int = 0
        self.end_char: int = 0
        self.__dict__.update(kwargs)

    @property
    def positions(self) -> bool:
        """ Whether tokens should have the token position in the
            'pos' attribute.
        """
        return self._positions

    @property
    def chars(self) -> bool:
        """ Whether tokens should have character offsets in the
            'start_char' and 'end_char' attributes.
        """
        return self._chars

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def remove_stopwords(self) -> bool:
        """ Whether to remove stop words from the stream (if
            the tokens pass through a stop filter).
        """
        return self._remove_stopwords

    def __repr__(self) -> str:
        params = ", ".join("%s=%r" % (name, value)
                           for name, value in self.__dict__.items())
        return "%s(%s)" % (self.__class__.__name__, params)

    def copy(self) -> "Token":
        # This is faster than using the copy module
        return Token(**self.__dict__)


class Composable(object):
    """
    Represents a composable objects that lead with "tokens". For instance, all Filters are Composable.
    """
    is_morph = False

    # def __or__(self, other):
    #     from whooshy.analysis.analyzers import CompositeAnalyzer
    #
    #     if not isinstance(other, Composable):
    #         raise TypeError("%r is not composable with %r" % (self, other))
    #     return CompositeAnalyzer(self, other)

    def __repr__(self):
        attrs = ""
        if self.__dict__:
            attrs = ", ".join("%s=%r" % (key, value)
                              for key, value
                              in self.__dict__.items())
        return self.__class__.__name__ + "(%s)" % attrs

    def __or__(self, other):
        from whooshy.analysis.analyzers import CompositeAnalyzer

        if not isinstance(other, Composable):
            raise TypeError("%r is not composable with %r" % (self, other))
        return CompositeAnalyzer(self, other)

    def has_morph(self):
        return self.is_morph
