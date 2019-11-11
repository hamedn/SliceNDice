import os
import glob
import pickle
import pandas as pd
import logging
import math
import networkx as nx
import matplotlib.pyplot as plt

from consts import *


def create_directory(dir_path):
    try:
        os.makedirs(dir_path)
        logging.info('Created directory {}'.format(dir_path))
    except OSError:
        logging.info('Failed to create directory {}'.format(dir_path))


def delete_dir_files(dir_path):
    """
    Deletes files under dir_path/*
    """

    files = glob.glob(dir_path + '*')
    for f in files:
        os.remove(f)
    logging.info('Deleted files under {}'.format(dir_path))


def print_line_to_log_file(txt, random_seed, log_dir_path):
    """
    Prints a line to a specific log file.

    Keyword arguments:
    txt -- The string to print to a text file
    random_seed -- The specific log file to print to (str)
    """

    with open(log_dir_path + str(random_seed), "a+") as f:
        f.write(str(txt) + "\n")


def print_object_to_log_file(obj, random_seed, log_dir_path):
    """
            Prints a line to a specific log file.

            Keyword arguments:
            obj -- The object to write to a pickle file (object)
            random_seed -- The specific log file to print to (str)
    """

    with open(log_dir_path + str(random_seed) + '.p', 'wb') as f:
        pickle.dump(obj, f)


def is_stopword(word, stopword_list, min_attribute_length=0):
    """
    Checks if a word is a stopword.

    Keyword arguments:
    word -- The input word (string)
    min_attribute_length -- If the word is <= this many characters, automatically consider it a stopword (int) (default=1)
    stopword_list -- The list of stopwords to compare against (list)

    Returns:
    True if is stop word, False if not a stop word
    """
    compare = str(word).strip().lower()
    stopword_condition = len(compare) <= min_attribute_length or compare in stopword_list
    return stopword_condition


def process_stopwords_dict_from_file(file, all_views):
    """
    Given a file with stopwords on each line, assign these stopwords to all views.
    :param file: file with stopwords
    :param all_views: set of view identifiers
    :return: dictionary keyed by view identifiers, with value (set of stopwords)
    """
    sws = set()
    for line in open(file, 'r'):
        sws.add(line.strip().lower())
    stopword_dict = {}
    for view in all_views:
        stopword_dict[view] = sws
    logging.info('Read {} stopwords from file {}'.format(len(sws), file))
    return stopword_dict


def is_valid_attr_value(val):
    """
    Check validity of an attribute value (not null, and type str)
    :param val: val to check validity of
    :return: True if valid, False if not
    """
    return (not pd.isnull(val)) and (isinstance(val, str) or not math.isnan(val))


def build_edge_label_dict(nx_edge_attr):
    interm_dict = {}
    for key, val in nx_edge_attr.items():
        u_v = (key[0], key[1])
        if u_v not in interm_dict:
            interm_dict[u_v] = {}
        if val[0] not in interm_dict[u_v]:
            interm_dict[u_v][val[0]] = []
        interm_dict[u_v][val[0]].append(str(val[1]))
    return_dict = {}
    for node_pair, attr_dict in interm_dict.items():
        edge_str = ''
        for view, attrs in attr_dict.items():
            edge_str += '{}: {}'.format(view, ', '.join([attr[:15] for attr in attrs[:1]])) + '\n'
        return_dict[node_pair] = edge_str
    return return_dict







