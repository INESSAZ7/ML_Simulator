#!/usr/bin/env python
# coding: utf-8

def word_count(batch, count=None):
    '''modified func for count word'''
    if count is None:
        count = {}
    for text in batch:
        for word in text.split():
            count[word] = count.get(word, 0) + 1
    return count
