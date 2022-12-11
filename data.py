import json
import os

class Example(object):

    def __init__(self,
                 id,
                 context_sentence,
                 start_ending,
                 endings,
                 answers):
        self.swag_id = id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = endings
        self.answers = answers


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 answers

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'doc_len': doc_len,
                'ques_len': ques_len,
                'option_len': option_len
            }
            for _, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len in choices_features
        ]
        self.answers = answers


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
            tokens_a.pop(1)
        else:
            tokens_b.pop(1)


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    ool = 0
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]  # + start_ending_tokens

            ending_token = tokenizer.tokenize(ending)
            option_len = len(ending_token)
            ques_len = len(start_ending_tokens)

            ending_tokens = start_ending_tokens + ending_token

            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            # ending_tokens = start_ending_tokens + ending_tokens
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            doc_len = len(context_tokens_choice)
            if len(ending_tokens) + len(context_tokens_choice) >= max_seq_length - 3:
                ques_len = len(ending_tokens) - option_len

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # assert (doc_len + ques_len + option_len) <= max_seq_length
            if (doc_len + ques_len + option_len) > max_seq_length:
                print(doc_len, ques_len, option_len, len(context_tokens_choice), len(ending_tokens))
                assert (doc_len + ques_len + option_len) <= max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len))
        answers = example.answers
        features.append(
            InputFeatures(
                example_id=example.swag_id,
                choices_features=choices_features,
                answers=answers
            )
        )

    return features


def read_instanse(f):
    d = json.load(open(f, encoding='utf8'))
    return Example(d['id'], d['article'], d['questions'], d['options'], d['answers'])


def read_examples(path):
    high = []
    middle = []
    for root, dirs, files in os.walk(os.path.join(path, 'high')):
        for f in files:
            high.append(read_instanse(os.path.join(root, f)))
    for root, dirs, files in os.walk(os.path.join(path, 'middle')):
        for f in files:
            middle.append(read_instanse(os.path.join(root, f)))
    return high, middle


data_path = './RACE'
tokenizer = ''
train_high, train_middle = read_examples(os.path.join(data_path, 'train'))
dev_high, dev_middle = read_examples(os.path.join(data_path, 'dev'))
test_high, test_middle = read_examples(os.path.join(data_path, 'test'))
# train_high_f = convert_examples_to_features(train_high, tokenizer, 512)
