#!/bin/bash
#
#
# Copyright 2017 The Board of Trustees of The Leland Stanford Junior University
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#
# Author: Peng Qi
# Modified by John Hewitt
#

# Train sections
for i in `seq -w 2 21`; do
        cat /u/nlp/data/PTB3/treebank_3/parsed/mrg/wsj/$i/*.mrg
done > ptb3-wsj-train.trees

# Dev sections
for i in 22; do
        cat /u/nlp/data/PTB3/treebank_3/parsed/mrg/wsj/$i/*.mrg
done > ptb3-wsj-dev.trees

# Test sections
for i in 23; do
        cat /u/nlp/data/PTB3/treebank_3/parsed/mrg/wsj/$i/*.mrg
done > ptb3-wsj-test.trees

for split in train dev test; do
    echo Converting $split split...
    java -mx1g edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile ptb3-wsj-${split}.trees -checkConnected -basic -keepPunct -conllx > ptb3-wsj-${split}.conllx
done

