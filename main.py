import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from tensorflow.keras.layers import Dense

from data import read_examples
from preprocessing import dcmn_preprocess
from model import GateLayers, DCMN

DATA_PATH = './RACE'

BERT_URL = {'L-2_H-128': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
            'L-4_H-256': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2'}

BERT_PREPROCESSOR_URL = {'small': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'}

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_passage_type', default='high', choices=['middle', 'high'])
    parser.add_argument('--train_passage_size', type=int, default=1000)
    parser.add_argument('--bert', required=True, choices=['L-2_H-128', ['L-4_H-256']])
    parser.add_argument('--bert_size', type=str, default='small')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--chkpt_path', default='./trained_models/')
    
    args = parser.parse_args()
    
    # Load preprocessor & bert layer
    preprocessor = hub.load(BERT_PREPROCESSOR_URL[args.bert_size])
    bert_preprocess = hub.KerasLayer(preprocessor)
    bert = hub.load(BERT_URL[args.bert])
    bert_layer = hub.KerasLayer(bert)
    
    # Load data
    train_high, train_middle = read_examples(os.path.join(DATA_PATH, 'train'))
    test_high, test_middle = read_examples(os.path.join(DATA_PATH, 'test'))
    
    # Take those used for training
    if args.train_passage_type == 'high':
        train_examples = train_high[:args.train_passage_size]
        # Always use 100 passages for testing
        test_examples = test_high[:100]
    else:
        train_examples = train_middle[:args.train_passage_size]
        test_examples = test_middle[:100]
        
    # Process data for BERT
    train_data, train_answers = dcmn_preprocess(train_examples)
    test_data, test_answers = dcmn_preprocess(test_examples)
    
    answer_dict = {'A': tf.constant([1, 0, 0, 0]),
               'B': tf.constant([0, 1, 0, 0]),
               'C': tf.constant([0, 0, 1, 0]),
               'D': tf.constant([0, 0, 0, 1])}
    train_labels = tf.convert_to_tensor([answer_dict[a] for a in train_answers])
    test_labels = tf.convert_to_tensor([answer_dict[a] for a in test_answers])
    
    # Build model
    config = {'n_choices': 4,
              #Under current setup, must adjust to match bert embedding size
              'hidden_size': int(args.bert[-3:]), 
              'dropout': 0.1,
              'bert_preprocess': bert_preprocess,
              'bert': bert_layer,
              'gate_layer': GateLayers,
              'classifier': Dense(4, activation='softmax')}
    
    dcmn_model = DCMN(config)
    
    dcmn_model.compile(
        optimizer = tf.keras.optimizers.Adam(args.lr),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = [tf.keras.metrics.CategoricalAccuracy()]
    )
    
    # Train and save weights
    dcmn_model.fit(x = train_data, 
              y = train_labels, 
              batch_size = args.batch_size, 
              epochs = args.epochs)
    
    dcmn_model.save_weights(args.chkpt_path)
    
if __name__ == '__main__':
    main()