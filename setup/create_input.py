#!/usr/bin/env python
# create a sparse document term matrix
# takes in an article glob and a lexicon (create_lexicon.py)
# creates two files:
#   dtm.out: each line represents 1 document as a sparse
#            count of words (wordindex:count) for every word in doc
#   filepath.out: each line has the full file name for the document
#                 in the corresponding line of dtm.out
import argparse
import pandas
from gensim.corpora import Dictionary
from nltk import tokenize

def main():
  parser = argparse.ArgumentParser()
  # input should be a tsv where 0 is doc id and 1 is text
  parser.add_argument('--input_file', default="/usr1/home/anjalief/ACL2013_Personas/preprocess/MovieSummaries/plot_summaries.txt")
  parser.add_argument('--out_file', default="plot_summaries.out")
  parser.add_argument('--vocab_file', default="vocab.out")
  args = parser.parse_args()

  data = pandas.read_csv(args.input_file, sep='\t')
  tokenized_data = [tokenize.word_tokenize(t.lower()) for t in data.iloc[:,1]]
  dct = Dictionary(tokenized_data)

  # NOTE: most people use tf-idf here, we're just keeping top 50,000 terms
  dct.filter_extremes(no_below=5, no_above=0.95, keep_n=50000)

  bow_forms = [dct.doc2bow(x) for x in tokenized_data]

  # Write BOW forms (sparse document-term matrix)
  outfile = open(args.out_file, "w")
  for doc,doc_id in zip(bow_forms, data.iloc[:,0]):
      outfile.write("%d\t" % doc_id)
      for tup in doc:
          outfile.write("%d:%d\t" % tup)
      outfile.write("\n")
  outfile.close()

  # write vocab, ordered by id
  vocabfile = open(args.vocab_file, "w")
  for key in sorted(dct.token2id, key=dct.token2id.get):
      vocabfile.write(key + "\n")
  vocabfile.close()

if __name__ == '__main__':
  main()

