# Copyright 2021 Matt Chaput and Marcos Pontes. All rights reserved.
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

"""
This module contains classes for scoring (and sorting) search results.
author: matt chaput and marcos pontes
"""
from __future__ import division

from collections import defaultdict
from math import log, pi, sqrt
from functools import lru_cache
from typing import Union

import whoosh.query
from whoosh.compat import iteritems
from whoosh.minterm import exist_minterm, extract_minterm
from whoosh.weighting_schema import IDF, TF


# Base classes


class WeightingModel(object):
    """Abstract base class for scoring models. A WeightingModel object provides
    a method, ``scorer``, which returns an instance of
    :class:`whoosh.scoring.Scorer`.

    Basically, WeightingModel objects store the configuration information for
    the model (for example, the values of B and K1 in the BM25F model), and
    then creates a scorer instance based on additional run-time information
    (the searcher, the fieldname, and term text) to do the actual scoring.
    """

    use_final = False
    DOC_FREQUENCY_CACHE = 100

    def idf(self, searcher, fieldname, text, idf_schema: IDF = IDF.default):
        """Returns the inverse document frequency of the given term.
        """

        parent = searcher.get_parent()
        return idf_schema(parent, fieldname, text)

    @lru_cache(maxsize=DOC_FREQUENCY_CACHE)
    def tf(self, searcher, fieldname, docnum, tf_schema: TF = TF.frequency):
        """Returns the document frequencies of the given (doc, term) pair.
        """
        all_tf = {}

        matcher = searcher.vector_as("weight", docnum, fieldname)
        for term, freq in matcher:
            max_weight = searcher.term_info(fieldname, term).max_weight()
            all_tf[term] = tf_schema(freq, max_weight)

        return all_tf

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        """Returns an instance of :class:`whoosh.scoring.Scorer` configured
        for the given searcher, fieldname, and term text.
        """

        raise NotImplementedError(self.__class__.__name__)

    def final(self, searcher, docnum, score):
        """Returns a final score for each document. You can use this method
        in subclasses to apply document-level adjustments to the score, for
        example using the value of stored field to influence the score
        (although that would be slow).

        WeightingModel sub-classes that use ``final()`` should have the
        attribute ``use_final`` set to ``True``.

        :param searcher: :class:`whoosh.searching.Searcher` for the index.
        :param docnum: the doc number of the document being scored.
        :param score: the document's accumulated term score.

        :rtype: float
        """

        return score


class BaseScorer(object):
    """Base class for "scorer" implementations. A scorer provides a method for
    scoring a document, and sometimes methods for rating the "quality" of a
    document and a matcher's current "block", to implement quality-based
    optimizations.

    Scorer objects are created by WeightingModel objects. Basically,
    WeightingModel objects store the configuration information for the model
    (for example, the values of B and K1 in the BM25F model), and then creates
    a scorer instance.
    """

    def supports_block_quality(self):
        """Returns True if this class supports quality optimizations.
        """

        return False

    def score(self, matcher):
        """Returns a score for the current document of the matcher.
        """

        raise NotImplementedError(self.__class__.__name__)

    def max_quality(self):
        """Returns the *maximum limit* on the possible score the matcher can
        give. This can be an estimate and not necessarily the actual maximum
        score possible, but it must never be less than the actual maximum
        score.
        """

        raise NotImplementedError(self.__class__.__name__)

    def block_quality(self, matcher):
        """Returns the *maximum limit* on the possible score the matcher can
        give **in its current "block"** (whatever concept of "block" the
        backend might use). This can be an estimate and not necessarily the
        actual maximum score possible, but it must never be less than the
        actual maximum score.

        If this score is less than the minimum score
        required to make the "top N" results, then we can tell the matcher to
        skip ahead to another block with better "quality".
        """

        raise NotImplementedError(self.__class__.__name__)


# Scorer that just returns term weight

class WeightScorer(BaseScorer):
    """A scorer that simply returns the weight as the score. This is useful
    for more complex weighting models to return when they are asked for a
    scorer for fields that aren't scorable (don't store field lengths).
    """

    def __init__(self, maxweight):
        self._maxweight = maxweight

    def supports_block_quality(self):
        return True

    def score(self, matcher):
        return matcher.weight()

    def max_quality(self):
        return self._maxweight

    def block_quality(self, matcher):
        return matcher.block_max_weight()

    @classmethod
    def for_(cls, searcher, fieldname, text):
        ti = searcher.term_info(fieldname, text)
        return cls(ti.max_weight())


# Base scorer for models that only use weight and field length

class WeightLengthScorer(BaseScorer):
    """Base class for scorers where the only per-document variables are term
    weight and field length.

    Subclasses should override the ``_score(weight, length)`` method to return
    the score for a document with the given weight and length, and call the
    ``setup()`` method at the end of the initializer to set up common
    attributes.
    """

    def setup(self, searcher, fieldname, text):
        """Initializes the scorer and then does the busy work of
        adding the ``dfl()`` function and maximum quality attribute.

        This method assumes the initializers of WeightLengthScorer subclasses
        always take ``searcher, offset, fieldname, text`` as the first three
        arguments. Any additional arguments given to this method are passed
        through to the initializer.

        Note: this method calls ``self._score()``, so you should only call it
        in the initializer after setting up whatever attributes ``_score()``
        depends on::

            class MyScorer(WeightLengthScorer):
                def __init__(self, searcher, fieldname, text, parm=1.0):
                    self.parm = parm
                    self.setup(searcher, fieldname, text)

                def _score(self, weight, length):
                    return (weight / (length + 1)) * self.parm
        """

        ti = searcher.term_info(fieldname, text)
        if not searcher.schema[fieldname].scorable:
            return WeightScorer(ti.max_weight())

        self.dfl = lambda docid: searcher.doc_field_length(docid, fieldname, 1)
        self._maxquality = self._score(ti.max_weight(), ti.min_length())

    def supports_block_quality(self):
        return True

    def score(self, matcher):
        return self._score(matcher.weight(), self.dfl(matcher.id()))

    def max_quality(self):
        return self._maxquality

    def block_quality(self, matcher):
        return self._score(matcher.block_max_weight(),
                           matcher.block_min_length())

    def _score(self, weight, length):
        # Override this method with the actual scoring function
        raise NotImplementedError(self.__class__.__name__)


# WeightingModel implementations

# Debugging model

class DebugModel(WeightingModel):
    def __init__(self):
        self.log = []

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        return DebugScorer(searcher, fieldname, text, self.log)


class DebugScorer(BaseScorer):
    def __init__(self, searcher, fieldname, text, log):
        ti = searcher.term_info(fieldname, text)
        self._maxweight = ti.max_weight()

        self.searcher = searcher
        self.fieldname = fieldname
        self.text = text
        self.log = log

    def supports_block_quality(self):
        return True

    def score(self, matcher):
        fieldname, text = self.fieldname, self.text
        docid = matcher.id()
        w = matcher.weight()
        length = self.searcher.doc_field_length(docid, fieldname)
        self.log.append((fieldname, text, docid, w, length))
        return w

    def max_quality(self):
        return self._maxweight

    def block_quality(self, matcher):
        return matcher.block_max_weight()


# BM25F Model

def bm25(idf, tf, fl, avgfl, B, K1):
    # idf - inverse document frequency
    # tf - term frequency in the current document
    # fl - field length in the current document
    # avgfl - average field length across documents in collection
    # B, K1 - free paramters

    return idf * ((tf * (K1 + 1)) / (tf + K1 * ((1 - B) + B * fl / avgfl)))


class BM25F(WeightingModel):
    """Implements the BM25F scoring algorithm.
    """

    def __init__(self, B=0.75, K1=1.2, **kwargs):
        """

        >>> from whoosh import scoring
        >>> # Set a custom B value for the "content" field
        >>> w = scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)

        :param B: free parameter, see the BM25 literature. Keyword arguments of
            the form ``fieldname_B`` (for example, ``body_B``) set field-
            specific values for B.
        :param K1: free parameter, see the BM25 literature.
        """

        self.B = B
        self.K1 = K1

        self._field_B = {}
        for k, v in iteritems(kwargs):
            if k.endswith("_B"):
                fieldname = k[:-2]
                self._field_B[fieldname] = v

    def supports_block_quality(self):
        return True

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        if not searcher.schema[fieldname].scorable:
            return WeightScorer.for_(searcher, fieldname, text)

        if fieldname in self._field_B:
            B = self._field_B[fieldname]
        else:
            B = self.B

        return BM25FScorer(searcher, fieldname, text, B, self.K1, qf=qf)


class BM25FScorer(WeightLengthScorer):
    def __init__(self, searcher, fieldname, text, B, K1, qf=1):
        # IDF and average field length are global statistics, so get them from
        # the top-level searcher
        parent = searcher.get_parent()  # Returns self if no parent
        self.idf = parent.idf(fieldname, text)
        self.avgfl = parent.avg_field_length(fieldname) or 1

        self.B = B
        self.K1 = K1
        self.qf = qf
        self.setup(searcher, fieldname, text)

    def _score(self, weight, length):
        s = bm25(self.idf, weight, length, self.avgfl, self.B, self.K1)
        return s


# DFree model

def dfree(tf, cf, qf, dl, fl):
    # tf - term frequency in current document
    # cf - term frequency in collection
    # qf - term frequency in query
    # dl - field length in current document
    # fl - total field length across all documents in collection
    prior = tf / dl
    post = (tf + 1.0) / (dl + 1.0)
    invpriorcol = fl / cf
    norm = tf * log(post / prior)

    return qf * norm * (tf * (log(prior * invpriorcol))
                        + (tf + 1.0) * (log(post * invpriorcol))
                        + 0.5 * log(post / prior))


class DFree(WeightingModel):
    """Implements the DFree scoring model from Terrier.

    See http://terrier.org/
    """

    def supports_block_quality(self):
        return True

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        if not searcher.schema[fieldname].scorable:
            return WeightScorer.for_(searcher, fieldname, text)

        return DFreeScorer(searcher, fieldname, text, qf=qf)


class DFreeScorer(WeightLengthScorer):
    def __init__(self, searcher, fieldname, text, qf=1):
        # Total term weight and total field length are global statistics, so
        # get them from the top-level searcher
        parent = searcher.get_parent()  # Returns self if no parent
        self.cf = parent.frequency(fieldname, text)
        self.fl = parent.field_length(fieldname)

        self.qf = qf
        self.setup(searcher, fieldname, text)

    def _score(self, weight, length):
        return dfree(weight, self.cf, self.qf, length, self.fl)


# PL2 model

rec_log2_of_e = 1.0 / log(2)


def pl2(tf, cf, qf, dc, fl, avgfl, c):
    # tf - term frequency in the current document
    # cf - term frequency in the collection
    # qf - term frequency in the query
    # dc - doc count
    # fl - field length in the current document
    # avgfl - average field length across all documents
    # c -free parameter

    TF = tf * log(1.0 + (c * avgfl) / fl)
    norm = 1.0 / (TF + 1.0)
    f = cf / dc
    return norm * qf * (TF * log(1.0 / f)
                        + f * rec_log2_of_e
                        + 0.5 * log(2 * pi * TF)
                        + TF * (log(TF) - rec_log2_of_e))


class PL2(WeightingModel):
    """Implements the PL2 scoring model from Terrier.

    See http://terrier.org/
    """

    def __init__(self, c=1.0):
        self.c = c

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        if not searcher.schema[fieldname].scorable:
            return WeightScorer.for_(searcher, fieldname, text)

        return PL2Scorer(searcher, fieldname, text, self.c, qf=qf)


class PL2Scorer(WeightLengthScorer):
    def __init__(self, searcher, fieldname, text, c, qf=1):
        # Total term weight, document count, and average field length are
        # global statistics, so get them from the top-level searcher
        parent = searcher.get_parent()  # Returns self if no parent
        self.cf = parent.frequency(fieldname, text)
        self.dc = parent.doc_count_all()
        self.avgfl = parent.avg_field_length(fieldname) or 1

        self.c = c
        self.qf = qf
        self.setup(searcher, fieldname, text)

    def _score(self, weight, length):
        return pl2(weight, self.cf, self.qf, self.dc, length, self.avgfl,
                   self.c)


class Boolean(WeightingModel):
    def supports_block_quality(self):
        return True

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        ql = 1.0
        if query_context:
            ql = len(query_context.subqueries)
        return BooleanScorer(ql)


class BooleanScorer(BaseScorer):
    def __init__(self, ql):
        self.ql = ql

    def max_quality(self):
        return 1.0

    def block_quality(self, matcher):
        return 1.0

    def score(self, matcher):
        return 1.0 / self.ql if matcher.weight() > 0.0 else 0.0


class Probabilistic(WeightingModel):

    def supports_block_quality(self):
        return True

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        N = searcher.doc_count()
        ni = searcher.doc_frequency(fieldname, text)
        return ProbabilisticScorer(N, ni)


class ProbabilisticScorer(BaseScorer):

    def __init__(self, N, ni):
        self.N = N
        self.ni = ni

    def score(self, matcher):
        return log((self.N + 0.5) / (self.ni + 0.5))

    def max_quality(self):
        return 1.0

    def block_quality(self, matcher):
        return 1.0


class Frequency(WeightingModel):
    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        maxweight = searcher.term_info(fieldname, text).max_weight()
        return WeightScorer(maxweight)


class TF_IDF(WeightingModel):
    """
    Using the VSM classic approach (which vector=True for the queried field), the first query is quite
    slower because of the calculation of all normalization terms. The next queries will be as faster as
    the original whoosh approach.
    """
    tf_normalize = True

    def __init__(self, *, tf: Union[TF, str] = TF.frequency, idf: Union[IDF, str] = IDF.default):
        """

        >>> from whoosh import scoring
        >>> # You can set IDF and TF schemas
        >>> w = scoring.TF_IDF(tf = TF.frequency, idf=IDF.inverse_frequency)

        :param idf: free parameter, indicates the idf schema. See the Vector Space Model literature.
        :param tf: free parameter, indicates the tf schema. See the Vector Space Model literature.
        """
        self.idf_table = dict()
        self.tf_table = defaultdict(dict)
        self.query_terms = set()

        self._idf_schema = IDF.get(idf) if isinstance(idf, str) else idf
        self._tf_schema = TF.get(tf) if isinstance(tf, str) else tf

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        # IDF is a global statistic, so get it from the top-level searcher
        self.extract_idf_table(searcher, fieldname, query_context)
        # Global context is needed for TF
        self.query_terms = set()
        self.all_terms(query_context)

        return TFIDFScorer(self, fieldname, text, searcher, query_context, self._tf_schema)

    def extract_idf_table(self, searcher, fieldname, query_context):
        if query_context:
            if isinstance(query_context, whoosh.query.compound.CompoundQuery):
                for subquery in query_context:
                    self.extract_idf_table(searcher, fieldname, subquery)
            elif isinstance(query_context, whoosh.query.terms.Term):
                self.idf_table[query_context.text] = searcher.idf(fieldname, query_context.text,
                                                                  self._idf_schema)

    def all_terms(self, context):
        if context:
            if isinstance(context, whoosh.query.compound.CompoundQuery):
                for subquery in context:
                    self.all_terms(subquery)
            elif isinstance(context, whoosh.query.terms.Term):
                self.query_terms.add(context.text)

    def apply_tf_normalization(self, score, docnum):
        norm_tf = 0.0
        for text, f in self.tf_table[docnum].items():
            if text in self.query_terms:
                norm_tf += (f * self.idf_table.get(text, 1)) ** 2
        return score / sqrt(norm_tf) if norm_tf != 0 else 0.0

    def tf(self, searcher, fieldname, docnum, tf_schema: TF = TF.frequency, context=None):
        # slow: usar só no último caso
        all_tf = super(TF_IDF, self).tf(searcher, fieldname, docnum, tf_schema)

        if context and not self.query_terms:
            self.all_terms(context)

        filtered_terms = {term: freq for term, freq in all_tf.items() if term in self.query_terms}

        return filtered_terms


class TFIDFScorer(BaseScorer):
    """
    Basic formulation of TFIDF Similarity based on Lucene score function.
    """

    def __init__(self, weighting, fieldname, text, searcher, query_context, tf_schema):
        self._fieldname = fieldname
        self._searcher = searcher
        self._weighting = weighting
        self._text = text.decode(encoding="utf-8")
        self._context = query_context
        self.tf_schema = tf_schema
        self.idf_table = weighting.idf_table
        self.norm_idf = sqrt(sum([v ** 2 for k, v in self.idf_table.items() if k in weighting.query_terms]))

    def supports_block_quality(self):
        return True

    def score(self, matcher):
        if self._searcher.has_vector(matcher.id(), self._fieldname):
            return self._approach(matcher)
        else:
            return matcher.weight() * self.idf_table.get(self._text, 1)  # whoosh default definition

    def _approach(self, matcher):

        tf_term = self._tf_statistics(matcher)
        idf_term, norm_idf = self._idf_statistics()

        return tf_term * (idf_term ** 2) / norm_idf

    def _idf_statistics(self):
        idf = self.idf_table.get(self._text, 1)
        return idf, self.norm_idf

    def _tf_statistics(self, matcher):
        maxweight = self._searcher.term_info(self._fieldname, self._text).max_weight()
        tf_term = self.tf_schema(matcher.weight(), maxweight)

        self._weighting.tf_table[matcher.id()][self._text] = tf_term

        return tf_term

    def max_quality(self):
        idf = self.idf_table[self._text]  # idf global statistics
        max_weight = self._searcher.term_info(self._fieldname, self._text).max_weight()

        return max_weight * idf

    def block_quality(self, matcher):
        idf = self.idf_table[self._text]  # idf global statistics

        return matcher.block_max_weight() * idf


class BeliefNetwork(TF_IDF):
    tf_normalize = True

    def __init__(self, tf: Union[TF, str] = TF.frequency, idf: Union[IDF, str] = IDF.default):
        """

        >>> from whoosh import scoring
        >>> # You can set IDF and TF schemas
        >>> w = scoring.BeliefNetwork(tf = TF.frequency, idf=IDF.inverse_frequency)

        :param idf: free parameter, indicates the idf schema. See the Vector Space Model literature.
        :param tf: free parameter, indicates the tf schema. See the Vector Space Model literature.
        """
        super().__init__(tf=tf, idf=idf)

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        # IDF is a global statistic, so get it from the top-level searcher
        # self.extract_idf_table(searcher, fieldname, query_context)
        self.extract_idf_table(searcher, fieldname, query_context)

        # Global context is needed for TF
        self.query_terms = set()
        self.all_terms(query_context)

        return BeliefNetworkScorer(self, fieldname, text, searcher, query_context,
                                   self._tf_schema, qf)


class BeliefNetworkScorer(TFIDFScorer):
    """
    Basic formulation of BeliefNetwork Similarity.
    """

    def __init__(self, weighting, fieldname, text, searcher, query_context, tf_schema, qf=1):
        super().__init__(weighting, fieldname, text, searcher, query_context, tf_schema)
        self.qf = qf
        self.t = len(query_context.subqueries)

    def _approach(self, matcher):
        tf_term = self._tf_statistics(matcher)
        idf_term, norm_idf = self._idf_statistics()

        p_dj_k = tf_term * idf_term
        p_q_k = self.qf * idf_term / norm_idf if norm_idf != 0 else 0.0
        p_k = (1 / 2) ** self.t

        return p_dj_k * p_q_k * p_k


class ExtendedBoolean(TF_IDF):
    tf_normalize = False

    def __init__(self, p=3):
        """

        >>> from whoosh import scoring
        >>> # You can set p value
        >>> w = scoring.ExtendedBoolean(p=3)

        """
        super().__init__(tf=TF.normalized_frequency, idf=IDF.inverse_frequency)
        self.p = p

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        # IDF is a global statistic, so get it from the top-level searcher
        self.extract_idf_table(searcher, fieldname, query_context)

        # Global context is needed for TF
        self.query_terms = set()
        self.all_terms(query_context)

        return ExtendedBooleanScorer(self, fieldname, text, searcher, query_context, self._tf_schema,
                                     self.p)

    @staticmethod
    def apply_function(score, p, qry_type):
        from math import pow
        if qry_type == 'AND':
            return 1 - pow(score, 1 / p if p != 0 else 1)
        return pow(score, 1 / p if p != 0 else 1)


class ExtendedBooleanScorer(TFIDFScorer):
    """
    Basic formulation of ExtendedBoolean Similarity.
    """

    def __init__(self, weighting, fieldname, text, searcher, query_context, tf_schema, p):
        super().__init__(weighting, fieldname, text, searcher, query_context, tf_schema)
        self.p = p
        self.qry_type = 'OR' if isinstance(query_context, whoosh.query.compound.Or) else 'AND'
        self.t = len(query_context.subqueries)

    def _approach(self, matcher):
        tf_term = self._tf_statistics(matcher)
        idf_term, _ = self._idf_statistics()

        wij = tf_term * idf_term

        if self.qry_type == 'AND':
            return (1 - wij) ** self.p / self.t
        return wij ** self.p / self.t


class GeneralizedVSM(TF_IDF):
    tf_normalize = True

    def __init__(self, *, mdb: str = None, tf: Union[TF, str] = TF.frequency, idf: Union[IDF, str] = IDF.default):
        """

        >>> from whoosh import scoring
        >>> # You can set IDF and TF schemas
        >>> w = scoring.GeneralizedVSM(tf = TF.frequency, idf=IDF.inverse_frequency)

        :param idf: free parameter, indicates the idf schema. See the Vector Space Model literature.
        :param tf: free parameter, indicates the tf schema. See the Vector Space Model literature.
        :param mdb: Minterm db base path. Example: 'mycol', 'mycol/body', etc.
        """
        super().__init__(tf=tf, idf=idf)
        self.mdb = mdb

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        # IDF is a global statistic, so get it from the top-level searcher
        self.extract_idf_table(searcher, fieldname, query_context)

        self.query_terms = set()
        self.all_terms(query_context)

        return GeneralizedVSMScorer(self, fieldname, text, searcher, query_context, self._tf_schema, self._idf_schema,
                                    self.mdb)


class GeneralizedVSMScorer(TFIDFScorer):
    """
    Basic formulation of BeliefNetwork Similarity.
    """

    def __init__(self, weighting, fieldname, text, searcher, query_context, tf_schema, idf_schema, mdb):
        super().__init__(weighting, fieldname, text, searcher, query_context, tf_schema)
        self.idf_schema = idf_schema
        self.mdb = mdb

    def _approach(self, matcher):
        if exist_minterm():
            tf_term = self._tf_statistics(matcher)
            idf_term, norm_idf = self._idf_statistics()

            return (tf_term * (idf_term ** 2) / norm_idf if norm_idf != 0 else 0.0) * \
                   extract_minterm(self.mdb, self._text, matcher.id())
        else:
            return matcher.weight() * self.idf_table.get(self._text, 1)  # whoosh default definition


# Utility models


class Weighting(WeightingModel):
    """This class provides backwards-compatibility with the old weighting
    class architecture, so any existing custom scorers don't need to be
    rewritten.
    """

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        return self.CompatibilityScorer(searcher, fieldname, text, self.score)

    def score(self, searcher, fieldname, text, docnum, weight):
        raise NotImplementedError

    class CompatibilityScorer(BaseScorer):
        def __init__(self, searcher, fieldname, text, scoremethod):
            self.searcher = searcher
            self.fieldname = fieldname
            self.text = text
            self.scoremethod = scoremethod

        def score(self, matcher):
            return self.scoremethod(self.searcher, self.fieldname, self.text,
                                    matcher.id(), matcher.weight())


class FunctionWeighting(WeightingModel):
    """Uses a supplied function to do the scoring. For simple scoring functions
    and experiments this may be simpler to use than writing a full weighting
    model class and scorer class.

    The function should accept the arguments
    ``searcher, fieldname, text, matcher``.

    For example, the following function will score documents based on the
    earliest position of the query term in the document::

        def pos_score_fn(searcher, fieldname, text, matcher):
            poses = matcher.value_as("positions")
            return 1.0 / (poses[0] + 1)

        pos_weighting = scoring.FunctionWeighting(pos_score_fn)
        with myindex.searcher(weighting=pos_weighting) as s:
            results = s.search(q)

    Note that the searcher passed to the function may be a per-segment searcher
    for performance reasons. If you want to get global statistics inside the
    function, you should use ``searcher.get_parent()`` to get the top-level
    searcher. (However, if you are using global statistics, you should probably
    write a real model/scorer combo so you can cache them on the object.)
    """

    def __init__(self, fn):
        self.fn = fn

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        return self.FunctionScorer(self.fn, searcher, fieldname, text, qf=qf)

    class FunctionScorer(BaseScorer):
        def __init__(self, fn, searcher, fieldname, text, qf=1):
            self.fn = fn
            self.searcher = searcher
            self.fieldname = fieldname
            self.text = text
            self.qf = qf

        def score(self, matcher):
            return self.fn(self.searcher, self.fieldname, self.text, matcher)


class MultiWeighting(WeightingModel):
    """Chooses from multiple scoring algorithms based on the field.
    """

    def __init__(self, default, **weightings):
        """The only non-keyword argument specifies the default
        :class:`Weighting` instance to use. Keyword arguments specify
        Weighting instances for specific fields.

        For example, to use ``BM25`` for most fields, but ``Frequency`` for
        the ``id`` field and ``TF_IDF`` for the ``keys`` field::

            mw = MultiWeighting(BM25(), id=Frequency(), keys=TF_IDF())

        :param default: the Weighting instance to use for fields not
            specified in the keyword arguments.
        """

        self.default = default
        # Store weighting functions by field name
        self.weightings = weightings

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        w = self.weightings.get(fieldname, self.default)
        return w.scorer(searcher, fieldname, text, qf=qf)


class ReverseWeighting(WeightingModel):
    """Wraps a weighting object and subtracts the wrapped model's scores from
    0, essentially reversing the weighting model.
    """

    def __init__(self, weighting):
        self.weighting = weighting

    def scorer(self, searcher, fieldname, text, qf=1, query_context=None):
        subscorer = self.weighting.scorer(searcher, fieldname, text, qf=qf)
        return ReverseWeighting.ReverseScorer(subscorer)

    class ReverseScorer(BaseScorer):
        def __init__(self, subscorer):
            self.subscorer = subscorer

        def supports_block_quality(self):
            return self.subscorer.supports_block_quality()

        def score(self, matcher):
            return 0 - self.subscorer.score(matcher)

        def max_quality(self):
            return 0 - self.subscorer.max_quality()

        def block_quality(self, matcher):
            return 0 - self.subscorer.block_quality(matcher)
