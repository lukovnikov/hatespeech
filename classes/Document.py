#encoding:utf-8
'''
Created on Oct, 2014
python 2.7.3
@author: hadyelsahar
'''

import pandas as pd
import regex


# Data Structure to represent document words
class Document:

    def preprocess(self, text):

        text = Document.remove_elongation(text)
        text = Document.clean(text)
        # text = Document.tag(text)

        return text

    @staticmethod
    def remove_elongation(text):

        return regex.sub(r'(.)\1{3,}', r'\1\1', text, flags=regex.UNICODE)


    @staticmethod
    def clean(text):

        #removing extra spaces
        text = regex.sub(r'[\s\n]+', ' ', text, flags=regex.UNICODE)

        # todo : add more cleaning methods

        return text


    @staticmethod
    def tag(text):

        # todo : adding tags to tag a dataset
        return text