#!/bin/bash

#script that computes the observed coherence (pointwise mutual information, normalised pmi or log 
#conditional probability)
#steps:
#1. sample the word counts of the topic words based on the reference corpus
#2. compute the observed coherence using the chosen metric

#parameters
metric="pmi" #evaluation metric: pmi, npmi or lcp
ref_corpus_dir="corpusWikiId"
wordcount_file="Wc-oc.txt"

#setting and compute
topic_file="coba100.txt"
oc_file="hasilcoba100.txt"
echo "Computing word occurrence..."
python ComputeWordCount.py $topic_file $ref_corpus_dir > $wordcount_file
echo "Computing the observed coherence..."
python ComputeObservedCoherence.py $topic_file $metric $wordcount_file > $oc_file
