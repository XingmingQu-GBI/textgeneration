from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from . import textgenrnn
from datetime import datetime
import os
import subprocess
import tensorflow as tf
import argparse
import sys

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")


def get_args():
    """Argument parser.
  Returns:
    Dictionary of arguments.
  """

    parser = argparse.ArgumentParser()

    parser.add_argument('--job-dir', type=str, required=True,
                        help='local or GCS location for writing checkpoints and exporting models'
                        )

    parser.add_argument('--data-dir', type=str, required=True,
                        help='local or GCS location for training txt file'
                        )

    parser.add_argument('--current-class', type=str, required=True,
                        help='local or GCS location for training txt file'
                        )

    parser.add_argument('--num-epochs', type=int, default=5,
                        help='number of times to go through the data, default=20'
                        )

    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of records to read during each training step, default=128'
                        )

#     parser.add_argument('--learning-rate', default=.01, type=float,
#                         help='learning rate for gradient descent, default=.01'
#                         )

    parser.add_argument('--verbosity', choices=['DEBUG', 'ERROR',
                                                'FATAL', 'INFO', 'WARN'], default='INFO')
    (args, _) = parser.parse_known_args()
    return args


def train_and_evaluate(args):
    """Trains and evaluates the Keras model.
  Uses the Keras model defined in model.py and trains on data loaded and
  preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
  format to the path defined in part by the --job-dir argument.
  Args:
    args: dictionary of arguments - see get_args() for details
  """

    file_name = args.data_dir
    # change to set file name of resulting trained models/texts
    model_name = args.current_class
    epoch = args.num_epochs
    batchSize = args.batch_size
    data_name = model_name+'.txt'
    
    print(model_name)
    subprocess.check_call(['gsutil', 'cp', file_name,
                           data_name], stderr=sys.stdout)
    
    subprocess.check_call(['gsutil', 'cp', 'gs://enrich_xingming/txtData/textgenrnn_vocab.json',
                           'textgenrnn_vocab.json'], stderr=sys.stdout)

    subprocess.check_call(['gsutil', 'cp', 'gs://enrich_xingming/txtData/textgenrnn_weights.hdf5',
                           'textgenrnn_weights.hdf5'], stderr=sys.stdout)
    
    model_cfg = {
        # set to True if want to train a word-level model (requires more data and smaller max_length)
        'word_level': False,
        # number of LSTM cells of each layer (128/256 recommended)
        'rnn_size': 128,
        'rnn_layers': 3,   # number of LSTM layers (>=2 recommended)
        # consider text both forwards and backward, can give a training boost
        'rnn_bidirectional': True,
        # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
        'max_length': 40,
        # maximum number of words to model; the rest will be ignored (word-level model only)
        'max_words': 20000,
    }

    train_cfg = {
        'line_delimited': True,   # set to True if each text has its own line in the source file
        'num_epochs': epoch,   # set higher to train the model for longer
        'gen_epochs': 2,   # generates sample text from model after given number of epochs
        # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
        'train_size': 0.9,
        'dropout': 0.2,   # ignore a random proportion of source tokens each epoch, allowing model to generalize better
        # If train__size < 1.0, test on holdout dataset; will make overall training slower
        'validation': False,
        'is_csv': False   # set to True if file is a CSV exported from Excel/BigQuery/pandas
    }

    textgen = textgenrnn(name=model_name,weights_path='textgenrnn_weights.hdf5',vocab_path='textgenrnn_vocab.json')

    train_function = textgen.train_from_file if train_cfg[
        'line_delimited'] else textgen.train_from_largetext_file

    train_function(
        file_path=data_name,
        new_model=True,
        num_epochs=train_cfg['num_epochs'],
        gen_epochs=train_cfg['gen_epochs'],
        batch_size=batchSize,
        train_size=train_cfg['train_size'],
        dropout=train_cfg['dropout'],
        validation=train_cfg['validation'],
        is_csv=train_cfg['is_csv'],
        rnn_layers=model_cfg['rnn_layers'],
        rnn_size=model_cfg['rnn_size'],
        rnn_bidirectional=model_cfg['rnn_bidirectional'],
        max_length=model_cfg['max_length'],
        dim_embeddings=64,
        word_level=model_cfg['word_level'])

      
    print("\n\nUploadin Models")  
    subprocess.check_call(['gsutil', 'cp', '{}_weights.hdf5'.format(model_name), args.job_dir],
        stderr=sys.stdout)
    subprocess.check_call(['gsutil', 'cp', '{}_vocab.json'.format(model_name), args.job_dir],
        stderr=sys.stdout)
    subprocess.check_call(['gsutil', 'cp', '{}_config.json'.format(model_name), args.job_dir],
        stderr=sys.stdout)

    print("\n\nfinished")


if __name__ == '__main__':

    args = get_args()
    tf.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)




