
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import *
from nltk.stem.porter import PorterStemmer
from operator import add
import re, string, unicodedata
import inflect
import hashlib
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

'''
1.  read all files in 1 RDD:  (filename,Entire_text_from_file)
2.  convert each (k,v) pair from unicode to String from RDD
3.  convert the v part to lowercase
5.  tokenize v into list of words
6.  remove stop words
7.  Create a tuple with count 1 ((word_id, doc_id)), 1)
8.  Group all (word_id, doc_id) pairs and sum the counts
9.  Transform tuple into (word_id, doc_id)
10. Group by word_ids and create list of grouped doc_ids in ascending order
11. Write the rdd into file sorted by word_ids.

'''
APP_NAME = " InvertedIndex"

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def uni_to_clean_str(text):
    converted_str = text.encode('utf-8')
    return converted_str.lower()


def tokenize_to_words(tuple):
    # split into words
    tokens = word_tokenize(tuple[1].lower().decode('utf-8'))
    stemmed_words = stemmer(tokens)
    words = remove_stopwords(stemmed_words)
    final = map(lambda x:(tuple[0], x), words)
    return final


def remove_stopwords(words):
    # filter out stop words
    stop_words_list = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words_list]
    return words


# stemming of words
def stemmer(tokens):
    porter = PorterStemmer()
    stemmed = [porter.stem(word).encode('utf-8') for word in tokens]
    return stemmed


def stemmer_word(word):
    # stemming of words
    porter = PorterStemmer()
    return porter.stem(word).encode('utf-8')


def create_word_to_doc_map(pair):
    l1 = list(pair[1])
    l1.sort()
    new_pair = (pair[0], l1)
    return new_pair


# Start buildIndex Function
def buildInvertedIndex(sc, inputFiles):
    inputFiles_rdd = sc.wholeTextFiles(inputFiles)

    # assign an unique id to each document file read and write it into doc_dictionary directory.
    documents_with_id_rdd = inputFiles_rdd.keys().zipWithIndex()
    documents_with_id_rdd.coalesce(1).saveAsTextFile("../../inverted-index/result/docs_dictionary")

    text_with_doc_id = inputFiles_rdd.join(documents_with_id_rdd).map(lambda x: (x[1][1],x[1][0]))

    inputFiles_rdd_partitioned = text_with_doc_id.repartition(4);

    # convert each fileName and its text from UNICODE to string, tokenize and delete all stopwords
    unicode_to_str_rdd = inputFiles_rdd_partitioned.map(
        lambda xy: (xy[0], uni_to_clean_str(xy[1])))

    #   by doc id tuple of (doc_id, word)
    tokenized_input_rdd_by_doc_id = unicode_to_str_rdd.flatMap(lambda xy: (tokenize_to_words(xy)))

    #   doc_id by words i.e. tuple (word, doc_id)
    tokenized_input_rdd_by_word = tokenized_input_rdd_by_doc_id.map(lambda x:(x[1],x[0]))

    #   picked up unique words and assign word_id
    words_by_id = tokenized_input_rdd_by_word.keys().distinct().zipWithIndex()
    #   words and word_id i.e. tuple(word, word_id)
    words_by_id.coalesce(1).saveAsTextFile("../../inverted-index/result/word_dictionary")

    #   Create a tuple with count 1 ((word_id, doc_id)), 1)
    #   Group all (word_id, doc_id) pairs and sum the counts
    #   Transform tuple into (word_id, doc_id)
    #   Group by word_ids and create list of grouped doc_ids in ascending order
    #   Write the rdd into file sorted by word_ids.
    words_id_by_doc_id = tokenized_input_rdd_by_word.join(words_by_id).map(lambda x:(x[1][1],x[1][0])).map(lambda x: (x,1)).reduceByKey(add)\
        .map(lambda x:(x[0][0],x[0][1])).groupByKey().map(lambda x : create_word_to_doc_map(x))
    # print(words_id_by_doc_id.take(100))
    words_id_by_doc_id.sortByKey().coalesce(1).saveAsTextFile("../../inverted-index/result/inverted_index")


# Configuration file
if __name__ == "__main__":
    # Configuration for Spark
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[8]")
    sc = SparkContext(conf=conf)
    inputFiles = "../../inverted-index/data"
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

    buildInvertedIndex(sc, inputFiles)
