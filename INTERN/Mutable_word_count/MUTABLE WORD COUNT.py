#!/usr/bin/env python
# coding: utf-8


import utils


def test_word_count():
    '''test for new and old fuct; both success'''
    some_text = ['мама мыла раму']
    test_dict = {'мама': 1, 'мыла': 1, 'раму': 1}
    count = utils.word_count(some_text)
    assert count == test_dict


def test_word_count_tricky():
    '''test for new and old fuct; new failed'''
    some_text = ['мама мыла раму']
    test_dict = {'мама': 1, 'мыла': 1, 'раму': 1}
    count = utils.word_count(some_text)
    count = utils.word_count(some_text)
    assert count == test_dict
