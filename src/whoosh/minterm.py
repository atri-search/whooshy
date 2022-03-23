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
Minterm for term correlated based algorithms.
"""
from collections import namedtuple, defaultdict
from functools import lru_cache
from math import sqrt
from typing import List, Set, Dict

from whoosh.weighting_schema import IDF, TF

Correlation = namedtuple("Correlation", "keyword cir")
Minterm = namedtuple("Minterm", "correlations id")


def __build_term_representation(minterm_db, minterms):
    term_repr = defaultdict(lambda: [0.0] * len(minterms))
    normalization = defaultdict(float)
    count = 0
    for minterm in sorted(minterms, key=lambda x: x.id):
        for correlation in minterm.correlations:
            term_repr[correlation.keyword][count] = correlation.cir
            normalization[correlation.keyword] += correlation.cir ** 2
        count += 1

    for term in term_repr:
        norm = normalization[term] if normalization[term] != 0 else 1.0
        term_repr[term] = [score / sqrt(norm) if sqrt(norm) != 0 else 0.0 for score in term_repr[term]]
        minterm_db.set(term, term_repr[term])


def __build_document_to_minterm_mapping(docs_to_minterm, minterms, document_term_correlations):
    mtid = 0
    for doc, correlations in document_term_correlations.items():
        flag = True
        for minterm in minterms:
            if {correlation.keyword for correlation in minterm.correlations} == \
                    {correlation.keyword for correlation in correlations}:
                flag = False
                new_correlations = set()
                for act_cor in minterm.correlations:
                    for oth_cor in correlations:
                        if oth_cor.keyword == act_cor.keyword:
                            new_correlations.add(Correlation(act_cor.keyword, act_cor.cir + oth_cor.cir))
                minterms[minterms.index(minterm)] = Minterm(new_correlations, minterm.id)
                docs_to_minterm.set(doc, minterm.id)
        if flag:
            minterms.append(Minterm(correlations, mtid))
            docs_to_minterm.set(doc, mtid)
            mtid += 1


def __build_term_correlations_by_document(all_docs, searcher, fieldname, tf_schema, idf_schema,
                                          doc_minterm: Dict[int, Set[Correlation]]):
    for docnum in all_docs:
        matcher = searcher.vector_as("weight", docnum, fieldname)
        for term, freq in matcher:

            max_weight = searcher.term_info(fieldname, term).max_weight()
            tf = tf_schema(freq, max_weight)

            idf = searcher.idf(fieldname, term, idf_schema)

            if docnum not in doc_minterm:
                doc_minterm[docnum] = set()

            doc_minterm[docnum].add(Correlation(term, tf * idf))


def exist_minterm() -> bool:
    try:
        import diskcache
        return True
    except ImportError:
        return False


@lru_cache(maxsize=None)
def extract_minterm(mdb, term, match_id):
    try:
        import diskcache as dc

        docs_to_minterm = dc.Cache("tmp/{}/docs_to_minterm".format(mdb))
        minterm_db = dc.Cache("tmp/{}/minterm_db".format(mdb))

        minterm = minterm_db.get(term)
        idx = docs_to_minterm.get(match_id)

        # convert bytes to dict
        if not (minterm and idx):
            raise KeyError(f"Minterm for {term} not found.")

        return minterm[idx]

    except ImportError:
        raise Exception("Diskcache is required to use Minterms.extract()")


def index_minterms(mdb, fieldname, ix, tf_schema=TF.frequency, idf_schema=IDF.inverse_frequency):
    """
    Build minterms for a certain fieldname. It must be executed after all indexing.
    """
    try:
        import diskcache as dc

        docs_to_minterm = dc.Cache("tmp/{}/docs_to_minterm".format(mdb))  # Dict[int, int]
        minterm_db = dc.Cache("tmp/{}/minterm_db".format(mdb))

        minterms: List[Minterm] = []
        document_term_correlations: Dict[int, Set[Correlation]] = dict()

        with ix.searcher() as searcher:

            doc_ids = searcher.ixreader.all_doc_ids()

            # build document_term_correlations
            __build_term_correlations_by_document(doc_ids, searcher, fieldname, tf_schema, idf_schema,
                                                  document_term_correlations)

            # build docs_to_minterm and minterms
            __build_document_to_minterm_mapping(docs_to_minterm, minterms, document_term_correlations)

            # build minterm_db
            __build_term_representation(minterm_db, minterms)

    except ImportError:
        raise Exception("Diskcache is required to use Minterms.index()")
