"""
    Data loading methods
"""
from collections import defaultdict
import csv
import numpy as np

from constants import *

class Batch:
    def __init__(self):
        self.docs = []
        self.labels = []
        self.hadm_ids = []
        self.max_length = 0 # Max length of doc in current batch
        self.cutoff_length = 3000

    def add_instance(self, row, w2ind):
        """
            Makes an instance to add to this batch from given row data, with a bunch of lookups
        """
        #labels = set()
        hadm_id = int(row[1])
        text = row[2] # SHOULD BE 2 WHEN CHARTTIME IS REMOVED
        label = int(row[3])

        #OOV words are given a unique index at end of vocab lookup
        text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()]

        # ADDED TO REPLACE AMBIGUOUS LENGTH ASSIGNMENT ABOVE
        length = len(text)
        #length = int(row[4]) # WHERE DOES THIS COME FROM?

        #reset length
        self.max_length = max(self.max_length, length) # NEED TO PAD TO EITHER MAX LENGTH OR LONGEST DOC LENGTH (?)

        #truncate long documents
        if len(text) > self.cutoff_length:
            text = text[:self.cutoff_length]

        #build instance
        self.docs.append(text)
        self.labels.append(label)
        self.hadm_ids.append(hadm_id)


    def pad_docs(self):
        #pad all docs to have self.length
        padded_docs = []
        final_max_length = min(self.max_length, self.cutoff_length) # Either cutoff_length or length of longest doc < cutoff_length
        for doc in self.docs:
            if len(doc) < final_max_length:
                doc.extend([0] * (final_max_length - len(doc)))
            padded_docs.append(doc)
        self.docs = self.docs

    def to_ret(self):
        return np.array(self.docs), np.array(self.labels), np.array(self.hadm_ids) #, self.code_set, np.array(self.descs)

def data_generator(filename, dicts, batch_size):

    """
        Inputs:
            filename: holds data sorted by sequence length, for best batching
            dicts: holds all needed lookups
            batch_size: the batch size for train iterations

        Output:
            Batch containing np array of data for training loop.
    """
    ind2w = dicts[0]
    w2ind = dicts[1]

    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r) # ERROR HANDLING WHEN ON LAST ROW?
        cur_inst = Batch()
        for row in r:
            #find the next `batch_size` instances
            if len(cur_inst.docs) == batch_size:
                cur_inst.pad_docs()
                yield cur_inst.to_ret()
                # CREATE NEW BATCH INSTANCE
                cur_inst = Batch()
            #cur_inst.add_instance(row, ind2c, c2ind, w2ind, dv_dict, num_labels)
            cur_inst.add_instance(row, w2ind) # HAVE TO CHECK if we need to return w2ind
        cur_inst.pad_docs()
        yield cur_inst.to_ret()


def load_vocab_dict(vocab_file): # SHOULD CHANGE TO load_vocab_dicts*

    ''' Input: Path to vocabulary file.

        Output: Index:Word dictionary, Word:Index dictionary'''

    #reads vocab_file into two lookups
    ind2w = defaultdict(str)
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                ind2w[i+1] = line.rstrip()
    w2ind = {w:i for i,w in ind2w.items()} # CHANGED FROM iteritems --> items
    return ind2w, w2ind

