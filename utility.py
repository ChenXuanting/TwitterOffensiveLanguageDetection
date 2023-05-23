import csv
import random
import regex
import os

from io import open
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
from abc import ABC, abstractmethod



from tqdm.autonotebook import tqdm

DataSet = List[Tuple[str, str, str]]

class DatasetHandle(object):

    def __init__(self, data: DataSet):
        self.ids, self.texts, self.labels = zip(*data)



class AbstractDataset(ABC):
    def __init__(self, *args, path: str, n_max : int = -1, shuffle=True, **kwargs):
        self.dataset: DataSet = None
        self.classes = set()
        self.n_max = n_max
        self.load(path)
        if shuffle:
          random.seed(9721)
          random.shuffle(self.dataset)
        _, _, labels = zip(*self.dataset)
        super().__init__()
    
    @abstractmethod
    def load(self, path: str):
        pass


class SubtaskAData(AbstractDataset):
    def load(self, path: str):

        # combine training and test data from OE2019 task to one training dataset
        # use it in 10-fold CV scenario
        # predict on OE2020 test data

        path19 = 'datasets/OffensEval19'
        off_data: DataSet = []
        with open(os.path.join(path19, "offenseval-training-v1.tsv"), 'r', encoding='UTF-8') as f:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                if i == self.n_max:
                    break
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                instance = (i, tweet, items[2])
                off_data.append(instance)
                self.classes.add(items[2])
        self.dataset = off_data

        testset19 = []
        with open(os.path.join(path19, "labels-levela.csv"), 'r', encoding='UTF-8') as f, open(os.path.join(path19, "testset-levela.tsv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(g)
            # read instances
            for i, line in enumerate(g):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = f.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (i, tweet, label[1])
                testset19.append(instance)
        self.dataset += testset19

        # load OE2020 test data
        self.testset = []
        with open(os.path.join(path, "test_a_tweets.tsv"), 'r', encoding='UTF-8') as f, open(os.path.join(path, "englishA-goldlabels.csv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = g.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (items[0], tweet, label[1])
                self.testset.append(instance)


    def getData(self) -> Tuple[DatasetHandle, DatasetHandle]:
        train_sentences = self.dataset
        test_sentences = self.testset
        return DatasetHandle(train_sentences), DatasetHandle(test_sentences)

class SubtaskBData(AbstractDataset):
    def load(self, path: str):

        # combine training and test data from OE2019 task to one training dataset
        # use it in 10-fold CV scenario
        # predict on OE2020 test data

        path19 = 'datasets/OffensEval19'
        off_data: DataSet = []
        with open(os.path.join(path19, "offenseval-training-v1.tsv"), 'r', encoding='UTF-8') as f:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                if i == self.n_max:
                    break
                items = line.split('\t')

                if items[3] == "NULL":
                    continue

                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                instance = (i, tweet, items[3])
                off_data.append(instance)
                self.classes.add(items[3])
        self.dataset = off_data

        testset19 = []
        with open(os.path.join(path19, "labels-levelb.csv"), 'r', encoding='UTF-8') as f, open(os.path.join(path19, "testset-levelb.tsv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(g)
            # read instances
            for i, line in enumerate(g):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = f.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (i, tweet, label[1])
                testset19.append(instance)
        self.dataset += testset19

        # load OE2020 test data
        self.testset = []
        with open(os.path.join(path, "test_b_tweets.tsv"), 'r', encoding='UTF-8') as f, open(os.path.join(path, "englishB-goldlabels.csv"), 'r', encoding='UTF-8') as g:
            # skip header
            next(f)
            # read instances
            for i, line in enumerate(f):
                items = line.split('\t')
                # remove URL and USER
                tweet = regex.sub("@USER ?|URL ?", "", items[1]).strip()
                label = g.readline().strip().split(",")
                assert label[0] == items[0], "IDs in line %d do not match" % i
                instance = (items[0], tweet, label[1])
                self.testset.append(instance)


    def getData(self, training_data_share = 0.9) -> Tuple[DatasetHandle, DatasetHandle]:
        train_sentences = self.dataset
        test_sentences = self.testset
        return DatasetHandle(train_sentences), DatasetHandle(test_sentences)

csv.field_size_limit(2147483647)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

   
    
def convert_example_to_feature(example_row, pad_token=0,
sequence_a_segment_id=0, sequence_b_segment_id=1,
cls_token_segment_id=1, pad_token_segment_id=0,
mask_padding_with_zero=True):
    example, label_map, max_seq_length, tokenizer, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)


    label_id = label_map[example.label]


    return InputFeatures(input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id)
    

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]',
                                 cls_token_segment_id=1, pad_token_segment_id=0):
    """ Loads a data file into a list of `InputBatch`s
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    examples = [(example, label_map, max_seq_length, tokenizer, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id) for example in examples]

    process_count = cpu_count() - 2

    with Pool(process_count) as p:
        features = list(tqdm(p.imap(convert_example_to_feature, examples, chunksize=100), total=len(examples)))


    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class BaseProcessor(object):
    """Processor for the sameside data set"""
    
    def __init__(self, trainset, devset):
        self.trainset = trainset
        self.devset = devset        

    def get_train_examples(self):
        return self._create_examples(
            self.trainset, "train")

    def get_dev_examples(self):
        return self._create_examples(
            self.devset, "dev")

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        
        for (i, item) in enumerate(items):
            guid = "%s-%s" % (set_type, i)
            text_a = item[0]
            text_b = item[1]
            label = item[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SubtaskAProcessor(BaseProcessor):

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        
        for (i, item) in enumerate(items):
            guid = "%s-%s" % (set_type, i)
            text_a = item[0]
            label = item[1]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples
    
    def get_labels(self):
        """See base class."""
        return ["NOT", "OFF"]
    

class SubtaskBProcessor(SubtaskAProcessor):
    
    def get_labels(self):
        """See base class."""
        return ["TIN", "UNT"]
    
class SubtaskCProcessor(SubtaskAProcessor):
    
    def get_labels(self):
        """See base class."""
        return ["IND", "GRP", "OTH"]
        

