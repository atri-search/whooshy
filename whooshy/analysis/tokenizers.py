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
from typing import AnyStr, Pattern, Dict, Union, Any

from whooshy.core import Composable, Token
from whooshy.util.text import rcompile, utf8decode

default_pattern: Pattern[AnyStr] = rcompile(r"\w+(\.?\w+)*")


def _assert_utf8(value):
    """
    Helper function that assert that all values are correctly encoded.
    :param value:
    :return:
    """
    if isinstance(value, bytes):
        value = utf8decode(value)

    assert isinstance(value, str), "%s is not unicode" % repr(value)
    return value


class Tokenizer(Composable):
    """Base class for Tokenizers.
    """

    def __eq__(self, other):
        return other and self.__class__ is other.__class__


class IDTokenizer(Tokenizer):
    """Yields the entire input string as a single token. For use in indexed but
    untokenized fields, such as a document's path.

    >>> idt = IDTokenizer()
    >>> [token.text for token in idt("/a/b 123 alpha")]
    ["/a/b 123 alpha"]
    """

    def __call__(self, value: AnyStr, positions: bool = False, chars: bool = False,
                 keep_original: bool = False, remove_stopwords: bool = True,
                 start_pos: int = 0, start_char: int = 0, mode: str = '', **kwargs):
        """
        :param value: utf8 bytes representation to be tokenized
        :param positions: Whether tokens should have the token position in the
            'pos' attribute.
        :param chars: Whether tokens should have character offsets in the
            'start_char' and 'end_char' attributes.
        :param keep_original: Whether the original text must be stored
        :param remove_stopwords: whether to remove stop words from the stream (if
            the tokens pass through a stop filter).
        :param start_pos: Start position of token
        :param start_char: Start char of token
        :param mode: contains a string describing the purpose for which the
            analyzer is being called, i.e. 'index' or 'query'.
        """
        value = _assert_utf8(value)

        t = Token(positions, chars, remove_stopwords=remove_stopwords, mode=mode,
                  **kwargs)

        t.text = value
        t.boost = 1.0

        if keep_original:
            t.original_text = value
        if positions:
            t.pos = start_pos + 1
        if chars:
            t.start_char = start_char
            t.end_char = start_char + len(value)
        yield t


class RegexTokenizer(Tokenizer):
    """
    Uses a regular expression to extract tokens from text.

    >>> rex = RegexTokenizer()
    >>> [token.text for token in rex(u("hi there 3.141 big-time under_score"))]
    ["hi", "there", "3.141", "big", "time", "under_score"]
    """

    def __init__(self, expression: Union[str, Any] = default_pattern, gaps: bool = False):
        """
        :param expression: A regular expression object or string. Each match
            of the expression equals a token. Group 0 (the entire matched text)
            is used as the text of the token. If you require more complicated
            handling of the expression match, simply write your own tokenizer.
        :param gaps: If True, the tokenizer *splits* on the expression, rather
            than matching on the expression.
        """

        self.expression = rcompile(expression)
        self.gaps = gaps

    def __eq__(self, other: "RegexTokenizer"):
        if self.__class__ is other.__class__:
            if self.expression.pattern == other.expression.pattern:
                return True
        return False

    def __call__(self, value: AnyStr, positions: bool = False, chars: bool = False, keep_original: bool = False,
                 remove_stopwords: bool = True, start_pos: int = 0, start_char: int = 0, tokenize: bool = True,
                 mode: str = '', **kwargs):
        """
        :param value: The unicode string to tokenize.
        :param positions: Whether to record token positions in the token.
        :param chars: Whether to record character offsets in the token.
        :param start_pos: The position number of the first token. For example,
            if you set start_pos=2, the tokens will be numbered 2,3,4,...
            instead of 0,1,2,...
        :param start_char: The offset of the first character of the first
            token. For example, if you set start_char=2, the text "aaa bbb"
            will have chars (2,5),(6,9) instead (0,3),(4,7).
        :param tokenize: if True, the text should be tokenized.
        """
        value = _assert_utf8(value)

        t = Token(positions, chars, remove_stopwords=remove_stopwords, mode=mode,
                  **kwargs)
        if not tokenize:
            t.original_text = t.text = value
            t.boost = 1.0
            if positions:
                t.pos = start_pos
            if chars:
                t.start_char = start_char
                t.end_char = start_char + len(value)
            yield t
        elif not self.gaps:
            # The default: expression matches are used as tokens
            for pos, match in enumerate(self.expression.finditer(value)):
                t.text = match.group(0)
                t.boost = 1.0
                if keep_original:
                    t.original_text = t.text
                t.stopped = False
                if positions:
                    t.pos = start_pos + pos
                if chars:
                    t.start_char = start_char + match.start()
                    t.start_char = start_char + match.end()
                yield t
        else:
            # When gaps=True, iterate through the matches and
            # yield the text between them.
            prevend = 0
            pos = start_pos
            for match in self.expression.finditer(value):
                start = prevend
                end = match.start()
                text = value[start:end]
                if text:
                    t.text = text
                    t.boost = 1.0
                    if keep_original:
                        t.original_text = t.text
                    t.stopped = False
                    if positions:
                        t.pos = pos
                        pos += 1
                    if chars:
                        t.start_char = start_char + start
                        t.end_char = start_char + end

                    yield t

                prevend = match.end()

            # If the last "gap" was before the end of the text,
            # yield the last bit of text as a final token.
            if prevend < len(value):
                t.text = value[prevend:]
                t.boost = 1.0
                if keep_original:
                    t.original_text = t.text
                t.stopped = False
                if positions:
                    t.pos = pos
                if chars:
                    t.start_char = prevend
                    t.end_char = len(value)
                yield t


class CharsetTokenizer(Tokenizer):
    """Tokenizes and translates text according to a character mapping object.
    Characters that map to None are considered token break characters. For all
    other characters the map is used to translate the character. This is useful
    for case and accent folding.

    This tokenizer loops character-by-character and so will likely be much
    slower than :class:`RegexTokenizer`.

    One way to get a character mapping object is to convert a Sphinx charset
    table file using :func:`whooshy.support.charset.charset_table_to_dict`.

    >>> from whooshy.support.charset import charset_table_to_dict
    >>> from whooshy.support.charset import default_charset
    >>> charmap = charset_table_to_dict(default_charset)
    >>> chtokenizer = CharsetTokenizer(charmap)
    >>> [t.text for t in chtokenizer(u'Stra\\xdfe ABC')]
    [u'strase', u'abc']

    The Sphinx charset table format is described at
    http://www.sphinxsearch.com/docs/current.html#conf-charset-table.
    """

    __inittype__ = dict(char_map=str)

    def __init__(self, char_map: Dict[int, chr]):
        """
        :param char_map: a mapping from integer character numbers to unicode
            characters, as used by the unicode.translate() method.
        """
        self.char_map = char_map

    def __eq__(self, other: "CharsetTokenizer"):
        return (other
                and self.__class__ is other.__class__
                and self.char_map == other.char_map)

    def __call__(self, value: AnyStr, positions: bool = False, chars: bool = False, keep_original: bool = False,
                 remove_stopwords: bool = True, start_pos: int = 0, start_char: int = 0, tokenize: bool = True,
                 mode: str = '', **kwargs):
        """
        :param value: The unicode string to tokenize.
        :param positions: Whether to record token positions in the token.
        :param chars: Whether to record character offsets in the token.
        :param start_pos: The position number of the first token. For example,
            if you set start_pos=2, the tokens will be numbered 2,3,4,...
            instead of 0,1,2,...
        :param start_char: The offset of the first character of the first
            token. For example, if you set start_char=2, the text "aaa bbb"
            will have chars (2,5),(6,9) instead (0,3),(4,7).
        :param tokenize: if True, the text should be tokenized.
        """
        value = _assert_utf8(value)

        t = Token(positions, chars, remove_stopwords=remove_stopwords, mode=mode, **kwargs)

        if not tokenize:
            t.original_text = t.text = value
            t.boost = 1.0
            if positions:
                t.pos = start_pos
            if chars:
                t.start_char = start_char
                t.end_char = start_char + len(value)
            yield t
        else:
            text = ""
            char_map = self.char_map
            pos = start_pos
            start = current = start_char
            for char in value:
                tchar = char_map[ord(char)]
                if tchar:
                    text += tchar
                else:
                    if current > start:
                        t.text = text
                        t.boost = 1.0
                        if keep_original:
                            t.original_text = t.text
                        if positions:
                            t.pos = pos
                            pos += 1
                        if chars:
                            t.start_char = start
                            t.end_char = current
                        yield t
                    start = current + 1
                    text = ""

                current += 1

            if current > start:
                t.text = value[start:current]
                t.boost = 1.0
                if keep_original:
                    t.original_text = t.text
                if positions:
                    t.pos = pos
                if chars:
                    t.start_char = start
                    t.end_char = current
                yield t


def SpaceSeparatedTokenizer() -> Tokenizer:
    """Returns a RegexTokenizer that splits tokens by whitespace.

    >>> sst = SpaceSeparatedTokenizer()
    >>> [token.text for token in sst("hi there big-time, what's up")]
    ["hi", "there", "big-time,", "what's", "up"]
    """
    return RegexTokenizer(r"[^ \t\r\n]+")


def CommaSeparatedTokenizer() -> Tokenizer:
    """Splits tokens by commas.

    Note that the tokenizer calls unicode.strip() on each match of the regular
    expression.

    >>> cst = CommaSeparatedTokenizer()
    >>> [token.text for token in cst("hi there, what's , up")]
    ["hi there", "what's", "up"]
    """

    from whooshy.analysis.filters import StripFilter

    return RegexTokenizer(r"[^,]+") | StripFilter()


class PathTokenizer(Tokenizer):
    """A simple tokenizer that given a string ``"/a/b/c"`` yields tokens
    ``["/a", "/a/b", "/a/b/c"]``.
    """

    def __init__(self, expression="[^/]+"):
        self.expr = rcompile(expression)

    def __call__(self, value, positions=False, start_pos=0, **kwargs):
        value = _assert_utf8(value)
        token = Token(positions, **kwargs)
        pos = start_pos
        for match in self.expr.finditer(value):
            token.text = value[:match.end()]
            if positions:
                token.pos = pos
                pos += 1
            yield token
