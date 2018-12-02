#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by YWJ on 2018/9/9

import pandas as pb

data = pb.read_csv("../data/train.csv", header=None,encoding='utf-8')

print(data.head(10))
print (data.shape)

datax = data.sort_values(by=0, ascending=False)


print (datax.shape)


survived  = data.groupby(5).size()
print (survived)
