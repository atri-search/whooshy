# coding=utf-8

from __future__ import with_statement

import pytest

from whooshy import analysis


def test_regextokenizer():
    value = "AAAaaaBBBbbbCCCcccDDDddd"

    rex = analysis.RegexTokenizer("[A-Z]+")
    assert [t.text for t in rex(value)] == ["AAA", "BBB", "CCC", "DDD"]

    rex = analysis.RegexTokenizer("[A-Z]+", gaps=True)
    assert [t.text for t in rex(value)] == ["aaa", "bbb", "ccc", "ddd"]


def test_path_tokenizer():
    value = "/alfa/bravo/charlie/delta/"
    pt = analysis.PathTokenizer()
    assert [t.text for t in pt(value)] == ["/alfa", "/alfa/bravo",
                                           "/alfa/bravo/charlie",
                                           "/alfa/bravo/charlie/delta"]


def test_composition1():
    ca = analysis.RegexTokenizer() | analysis.LowercaseFilter()
    assert ca.__class__.__name__ == "CompositeAnalyzer"
    assert ca[0].__class__.__name__ == "RegexTokenizer"
    assert ca[1].__class__.__name__ == "LowercaseFilter"
    assert [t.text for t in ca("ABC 123")] == ["abc", "123"]


def test_composition2():
    ca = analysis.RegexTokenizer() | analysis.LowercaseFilter()
    sa = ca | analysis.StopFilter()
    assert len(sa), 3
    assert sa.__class__.__name__ == "CompositeAnalyzer"
    assert sa[0].__class__.__name__ == "RegexTokenizer"
    assert sa[1].__class__.__name__ == "LowercaseFilter"
    assert sa[2].__class__.__name__ == "StopFilter"
    assert [t.text for t in sa("The ABC 123")], ["abc", "123"]


def test_composition3():
    sa = analysis.RegexTokenizer() | analysis.StopFilter()
    assert sa.__class__.__name__ == "CompositeAnalyzer"


def test_composing_functions():
    tokenizer = analysis.RegexTokenizer()

    def filter(tokens):
        for t in tokens:
            t.text = t.text.upper()
            yield t

    with pytest.raises(TypeError):
        tokenizer | filter


def test_multifilter():
    f1 = analysis.LowercaseFilter()
    f2 = analysis.PassFilter()
    mf = analysis.MultiFilter(a=f1, b=f2)
    ana = analysis.RegexTokenizer(r"\S+") | mf
    text = "ALFA BRAVO CHARLIE"
    assert [t.text for t in ana(text, mode="a")] == ["alfa", "bravo", "charlie"]
    assert [t.text for t in ana(text, mode="b")] == ["ALFA", "BRAVO", "CHARLIE"]