#!/usr/bin/python
# encoding: utf-8
import sys
import torch
import torch.nn as nn

from torch.autograd import Variable

import collections
import numpy as np


def get_all_korean():

    def nextKorLetterFrom(letter):
        lastLetterInt = 15572643
        if not letter:
            return '가'
        a = letter
        b = a.encode('utf8')
        c = int(b.hex(), 16)

        if c == lastLetterInt:
            return False

        d = hex(c + 1)
        e = bytearray.fromhex(d[2:])

        flag = True
        while flag:
            try:
                r = e.decode('utf-8')
                flag = False
            except UnicodeDecodeError:
                c = c+1
                d = hex(c)
                e = bytearray.fromhex(d[2:])
        return e.decode()

    returns = []
    flag = True
    k = ''
    while flag:
        k = nextKorLetterFrom(k)
        if k is False:
            flag = False
        else:
            returns.append(k)
    return returns


class strLabelConverterForAttention(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self):
        self.alphabet = get_all_korean()
        self.maxLen = -1
        
        self.dict = {}
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i

        self.dict['<EOS>'] = len(self.alphabet)

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        
        text_in = [self.dict[char] for char in text]
        length = [len(text_in)]
        # just for multi-GPU training, '[0]' is meaningless
        text_ex = text_in.copy()
        text_ex.extend([0]*(self.maxLen-len(text_ex)))

        return (torch.LongTensor(text_in), 
            torch.LongTensor(text_ex), 
            torch.IntTensor(length))


def lexicontoid(length,writerid):
    lexicon_writerID = torch.LongTensor(length.sum().item()).fill_(0)
    start = 0
    for i,len in enumerate(length):
        lexicon_writerID[start:start+len] = writerid[i].expand(len)
        start = start + len
    return lexicon_writerID


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self):
        self.length = [19,21,28]

    def encode(self, text):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """

        # length = [len(s.split()) for s in text]  # +1 for [s] at end of sentence.
        length = [3+2 for _ in text]
        return (text, torch.IntTensor(length))
