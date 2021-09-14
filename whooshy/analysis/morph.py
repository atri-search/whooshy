# Copyright 2021 Marcos Pontes. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY MARCOS PONTES ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL MARCOS PONTES OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Marcos Pontes.

from whooshy.analysis.filters import Filter

from whooshy.util import get_stemmer_by_lang


class StemFilter(Filter):
    """Stems (removes suffixes from) the text of tokens using the Porter
    stemming algorithm. Stemming attempts to reduce multiple forms of the same
    root word (for example, "rendering", "renders", "rendered", etc.) to a
    single word in the index.

    >>> stemmer = RegexTokenizer() | StemFilter()
    >>> [token.text for token in stemmer("fundamentally willows")]
    ["fundament", "willow"]

    You can pass the `lang` keyword argument.

    >>> stemfilter = StemFilter(lang="portuguese")

    By default, this class wraps an LRU cache around the stemming function. The
    ``cachesize`` keyword argument sets the size of the cache. To make the
    cache unbounded (the class caches every input), use ``cachesize=-1``. To
    disable caching, use ``cachesize=None``.

    If you compile and install the py-stemmer library, the
    :class:`PyStemmerFilter` provides slightly easier access to the language
    stemmers in that library.
    """

    __inittypes__ = dict(stemfn=object, ignore=list)

    is_morph = True

    def __init__(self, stemmer=None, lang="english", ignore=None):
        """
        :param stemmer: NLTK Stemmer classes
        :param lang: If stemmer is None, we'll set one for you based on language
        :param ignore:
        """
        self.lang = lang
        self.ignore = frozenset() if ignore is None else frozenset(ignore)
        self.stemmer = stemmer if stemmer else get_stemmer_by_lang(lang)

    def __call__(self, tokens):
        stemmer = self.stemmer
        ignore = self.ignore

        for t in tokens:
            if not t.stopped:
                text = t.text
                if text not in ignore:
                    t.text = stemmer.stem(text)
            yield t

