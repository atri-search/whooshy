# Copyright 2021 Marcos Pontes. All rights reserved.
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
# policies, either expressed or implied, of MARCOS PONTES.
"""
    Define different idf weighting schemas based on information retrieval literature.

    author: marcosfpr
"""

import enum
from math import log, sqrt

"""
All idf functions must accomplish the follow definition:
(searcher, fieldname, term) -> float
"""


def _default_idf(searcher, fieldname, term):
    """
        Default calculation of idf value of term given a field.
    """
    n = searcher.doc_frequency(fieldname, term)
    dc = searcher.doc_count_all()
    return log((dc + 1) / (n + 1)) + 1 if n + 1 != 0.0 else 0.0  # default whoosh schema


def _unary(searcher, fieldname, term):
    """
        Trivial unary idf calculation
    """
    return 1


def _inverse_frequency(searcher, fieldname, term):
    """
        Inverse frequency idf schema
    """
    n = searcher.doc_frequency(fieldname, term)
    dc = searcher.doc_count_all()
    return log(dc / n, 10) if n != 0.0 else 0.0


def _inverse_frequency_smooth(searcher, fieldname, term):
    """
        Inverse frequency smooth idf schema
    """
    n = searcher.doc_frequency(fieldname, term)
    dc = searcher.doc_count_all()
    return log(1 + (dc / n), 10) if n != 0.0 else 0.0


def _inverse_frequency_max(searcher, fieldname, term):
    """
        Inverse frequency smooth idf schema
    """
    n = searcher.doc_frequency(fieldname, term)
    maxweight = searcher.term_info(fieldname, term).max_weight()
    return log(1 + (maxweight / n), 10) if n != 0.0 else 0.0


def _probabilistic_inverse_frequency(searcher, fieldname, term):
    """
        Inverse frequency smooth idf schema
    """
    n = searcher.doc_frequency(fieldname, term)
    dc = searcher.doc_count_all()
    return log((dc - n) / n, 10) if n != 0.0 else 0.0


class IDF(enum.Enum):
    """
        Base class that represents all possible IDF schemas
    """
    default = _default_idf
    inverse_frequency = _inverse_frequency
    inverse_frequency_smooth = _inverse_frequency_smooth
    inverse_frequency_max = _inverse_frequency_max
    probabilistic_inverse_frequency = _probabilistic_inverse_frequency
    unary = _unary

    def __call__(self, *args):
        """
        Call the function with arguments
        """
        self.value(*args)


    @classmethod
    def get(cls, schema: str):
        """
           Get the IDF schema based on string values. The possibilities are:
           1. default
           2. inverse_frequency
           3. inverse_frequency_smooth
           4. inverse_frequency_max
           5. probabilistic_inverse_frequency
           6. unary
        """
        schema = schema.lower().strip()
        if schema == 'default':
            return cls.default
        elif schema == 'inverse_frequency':
            return cls.inverse_frequency
        elif schema == 'inverse_frequency_smooth':
            return cls.inverse_frequency_smooth
        elif schema == 'inverse_frequency_max':
            return cls.inverse_frequency_max
        elif schema == 'probabilistic_inverse_frequency':
            return cls.probabilistic_inverse_frequency
        raise NotImplementedError(f"No IDF schema called {schema} found.")


def _default(weight, max_weight):
    return sqrt(weight)


def _frequency(weight, max_weight):
    return weight


def _normalized_frequency(weight, max_weight):
    return weight / max_weight if max_weight != 0.0 else 0.0


def _log_normalization(weight, max_weight):
    return 1 + log(weight, 10)


def _double_normalization(weight, max_weight):
    return 0.5 + 0.5 * (weight / max_weight if max_weight != 0.0 else 0.0)


def _binary(weight, max_weight):
    return 1 if weight > 0 else 0


class TF(enum.Enum):
    """
        Base class that represents all possible TF schemas
    """
    default = _default
    frequency = _frequency
    normalized_frequency = _normalized_frequency
    log_normalization = _log_normalization
    double_normalization = _double_normalization
    binary = _binary

    def __call__(self, *args):
        """
        Call the function with arguments
        """
        self.value(*args)

    @classmethod
    def get(cls, schema: str):
        """
           Get the TF schema based on string values. The possibilities are:
           1. frequency
           2. log_normalization
           3. double_normalization
           4. binary
        """
        schema = schema.lower().strip()
        if schema == 'default':
            return cls.default
        elif schema == 'frequency':
            return cls.frequency
        elif schema == 'normalized_frequency':
            return cls.normalized_frequency
        elif schema == 'double_normalization':
            return cls.double_normalization
        elif schema == 'log_normalization':
            return cls.log_normalization
        elif schema == 'binary':
            return cls.binary
        raise NotImplementedError(f"No TF schema called {schema} found.")