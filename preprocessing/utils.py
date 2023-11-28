# -*- encoding: utf-8 -*-
'''
@Time       : 2022/11/26 17:19
@Author     : Tian Tan
@Email      :
@File       : compound.py
@Project    :
@Description:
'''


def one_of_k_encoding(x, allowable_set):
    # if x not in allowable_set:
    #     raise Exception("input {0} not in allowable set{1}:".format(
    #         x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]
