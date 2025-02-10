from classifier_lib import *

b = BayesClassifier()
a_list_of_words = ["I", "really", "like", "this", "movie", ".", "I", "hope", \
                    "you", "like", "it", "too"]
a_dictionary = {}
b.update_dict(a_list_of_words, a_dictionary)

def test_update_dict_1():
    assert a_dictionary["I"] == 2, "update_dict test 1"


def test_update_dict_2():
    assert a_dictionary["like"] == 2, "update_dict test 2"


def test_update_dict_3():
    assert a_dictionary["really"] == 1, "update_dict test 3"


def test_update_dict_4():
    assert a_dictionary["too"] == 1, "update_dict test 4"

def test_pos_dominator():
    assert sum(b.pos_freqs.values()) == 612445, "pos denominator test"


def test_neg_dominator():
    assert sum(b.neg_freqs.values()) == 129404, "neg denominator test"

def test_word_freqs1():
    assert b.pos_freqs['love'] == 1526, "word test 1"

def test_word_freqs2():
    assert b.neg_freqs['love'] == 99, "word test 2"

def test_word_freqs3():
    assert b.pos_freqs['terrible'] == 37, "word test 3"

def test_word_freqs4():
    assert b.neg_freqs['terrible'] == 176, "word test 4"

def test_word_freqs5():
    assert b.pos_freqs['the'] == 30590, "word test 5"

def test_word_freqs6():
    assert b.neg_freqs['the'] == 5646, "word test 6"

def test_word_freqs7():
    assert b.pos_freqs['computer'] == 46, "word test 7"

def test_word_freqs8():
    assert b.neg_freqs['computer'] == 12, "word test 8"

def test_classification1():
    assert b.classify('I love computer science') == "positive", "classify test 1"


def test_classification2():
    assert b.classify('this movie is fantastic') == "positive", "classify test 2"


def test_classification3():
    assert b.classify('great') == "positive", "classify test 3"


def test_classification4():
    assert b.classify('rainy days are the worst') == "negative", "classify test 4"


def test_classification5():
    assert b.classify('computer science is terrible') == "negative", "classify test 5"

def test_not_existent_in_corpus():
    #assuming that this token doesn't occur in either corpus, should default to negative
    #    because smaller denominator
    assert b.classify('blaaaaaaa') == "negative", "classify test 6"