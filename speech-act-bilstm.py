from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import math
import numpy as np
import tensorflow as tf
import optparse
import sys
import math
import glob, os, csv, re
from collections import Counter

from utilities import aidr_hierachical
from sklearn import metrics


def forward_propagation_bidirectional(word_ids, sequence_lengths, E):
    # embedding matrix
    E = tf.convert_to_tensor(E, tf.float32)
    W_embedding = tf.get_variable("W_embedding", initializer=E)

    print("Input data shape: ", word_ids.shape)
    data = tf.nn.embedding_lookup(W_embedding, word_ids)
    print("After word embedding input shape: ", data.shape)

    
    #put the time dimension (here words in sentence) on axis=1
    (_,dim_word)=np.shape(E)
    s=tf.shape(data)
    data = tf.reshape(data, [s[0] * s[1], s[2], dim_word])
    sequence_lengths=tf.reshape(sequence_lengths,[s[0]*s[1]])

    if options.recur_type=='lstm':
        cell_fw = tf.contrib.rnn.LSTMCell(options.hidden_size)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=options.dropout_ratio)
        cell_bw = tf.contrib.rnn.LSTMCell(options.hidden_size)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=options.dropout_ratio)

        (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, data, sequence_length=sequence_lengths, dtype=tf.float32)

        (c_fw, h_fw) = state_fw
        (c_bw, h_bw) = state_bw
        print("h_fw: ", h_fw.shape)
        print("h_bw: ", h_bw.shape)

        c = tf.concat([h_fw, h_bw], axis=-1)
        print("c: ", c.shape)

    elif options.recur_type=='gru':
        cell_fw = tf.contrib.rnn.GRUCell(options.hidden_size)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=options.dropout_ratio)
        cell_bw = tf.contrib.rnn.GRUCell(options.hidden_size)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=options.dropout_ratio)

        (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, data, sequence_length=sequence_lengths, dtype=tf.float32)

        print("state_fw: ", state_fw.shape)
        print("state_bw: ", state_bw.shape)

        c = tf.concat([state_fw, state_bw], axis=-1)
        print("c: ", c.shape)

    word_level_output = c

    weight = tf.get_variable("w", shape=[2*options.hidden_size, options.numClasses],
                             initializer=tf.contrib.layers.xavier_initializer(seed=101))
    bias = tf.get_variable("b", shape=[options.numClasses], initializer=tf.constant_initializer(0.0))
    prediction = (tf.matmul(word_level_output, weight) + bias)
    prediction = tf.reshape(prediction, [s[0], s[1], options.numClasses])

    return prediction


def mini_batches(X, Y, seq_len, num_sen, mini_batch_size=32, repeat_data=10):
    """
    Creates a list of minibatches from (X, Y)

    Arguments:
    X -- input data [3D shape (conv X num_sentences X maxlen)]
    Y -- label [2D array containing values 0-4 for 5 classes]
    seq_len -- 2D array containing number of words in sentence of conversation
    num_sen -- Number of sentences in each conversation
    mini_batch_size -- Size of each mini batch

    Returns:
    list of mini batches from the positive and negative documents.

    """
    m = X.shape[0]
    mini_batches = []

    num_complete_minibatches = int(math.floor(m / mini_batch_size))

    for k in range(0, num_complete_minibatches):
        start = (k * mini_batch_size)
        end = k * mini_batch_size + mini_batch_size
        mini_batch_X = X[start: end]
        mini_batch_Y = Y[start: end]
        # mini_batch_Y_one_hot = tf.one_hot(mini_batch_Y, numClasses)
        mini_batch_seqlen = seq_len[start: end]
        mini_batch_num_sen = num_sen[start: end]

        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_seqlen, mini_batch_num_sen)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    
    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = Y[num_complete_minibatches * mini_batch_size: m]
        # mini_batch_Y_one_hot = tf.one_hot(mini_batch_Y, numClasses)
        mini_batch_seqlen = seq_len[num_complete_minibatches * mini_batch_size: m]
        mini_batch_num_sen = num_sen[num_complete_minibatches * mini_batch_size: m]

        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_seqlen, mini_batch_num_sen)
        mini_batches.append(mini_batch)
    

    return mini_batches


if __name__ == '__main__':

    # parse user input
    parser = optparse.OptionParser("%prog [options]")

    # file related options
    parser.add_option("-g", "--log-file", dest="log_file", help="log file [default: %default]")
    parser.add_option("-d", "--data-dir", dest="data_dir",
                      help="directory containing train, test and dev file [default: %default]")
    parser.add_option("-D", "--data-spec", dest="data_spec",
                      help="specification for training data (in, out, in_out) [default: %default]")
    parser.add_option("-p", "--model-dir", dest="model_dir",
                      help="directory to save the best models [default: %default]")

    # network related
    parser.add_option("-t", "--max-tweet-length", dest="maxlen", type="int",
                      help="maximal tweet length (for fixed size input) [default: %default]")  # input size

    parser.add_option("-m", "--model-type", dest="model_type",
                      help="uni or bidirectional [default: %default]")  # uni, bi-directional
    parser.add_option("-r", "--recurrent-type", dest="recur_type",
                      help="recurrent types (lstm, gru, simpleRNN) [default: %default]")  # lstm, gru, simpleRNN
    parser.add_option("-v", "--vocabulary-size", dest="max_features", type="int",
                      help="vocabulary size [default: %default]")  # emb matrix row size
    parser.add_option("-e", "--emb-size", dest="emb_size", type="int",
                      help="dimension of embedding [default: %default]")  # emb matrix col size
    parser.add_option("-s", "--hidden-size", dest="hidden_size", type="int",
                      help="hidden layer size [default: %default]")  # size of the hidden layer
    parser.add_option("-o", "--dropout_ratio", dest="dropout_ratio", type="float",
                      help="ratio of cells to drop out [default: %default]")
    parser.add_option("-i", "--init-type", dest="init_type", help="random or pretrained [default: %default]")
    parser.add_option("-f", "--emb-file", dest="emb_file", help="file containing the word vectors [default: %default]")
    parser.add_option("-P", "--tune-emb", dest="tune_emb", action="store_false",
                      help="DON't tune word embeddings [default: %default]")
    parser.add_option("-z", "--num-class", dest="numClasses", type="int",
                      help="Number of output classes [default: %default]")
    parser.add_option("-E", "--eval-minibatches", dest="evalMinibatches", type="int",
                      help="After how many minibatch do we want to evaluate. [default: %default]")

    # learning related
    parser.add_option("-a", "--learning-algorithm", dest="learn_alg",
                      help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta) [default: %default]")
    parser.add_option("-L", "--learning-rate", dest="learning_rate", type="float",
                      help="learning rate of the optimizer [default: %default]")
    parser.add_option("-b", "--minibatch-size", dest="minibatch_size", type="int",
                      help="minibatch size [default: %default]")
    parser.add_option("-l", "--loss", dest="loss",
                      help="loss type (hinge, squared_hinge, binary_crossentropy) [default: %default]")
    parser.add_option("-n", "--epochs", dest="epochs", type="int", help="nb of epochs [default: %default]")
    parser.add_option("-C", "--map-class", dest="map_class", type="int",
                      help="map classes to five labels [default: %default]")
    parser.add_option("-F", "--fold", dest="fold", type="int",
                      help="denotes fold number. Here we are using 2 fold cross validation")


    parser.set_defaults(
        data_dir= "./data/new_data/New Speech-act Data/ta/"
        , data_spec="in"

        , model_dir="./saved_models/"
        , log_file="log"

        , learn_alg="adam"  # sgd, adagrad, rmsprop, adadelta, adam (default)
        , loss="softmax_crossentropy"  # hinge, squared_hinge, binary_crossentropy (default)
        , minibatch_size=5
        , dropout_ratio=0.75

        , maxlen=100
        , epochs=30
        , max_features=10000
        , emb_size=300
        , hidden_size=128
        , model_type='bidirectional'  # bidirectional, unidirectional (default)
        , recur_type='lstm'  # gru, simplernn, lstm (default)
        , init_type='conv_glove'  # 'random', 'word2vec', 'glove', 'conv_word2vec', 'conv_glove', 'meta_conv',  'meta_orig'
        , emb_file="../data/unlabeled_corpus.vec"
        , tune_emb=True
        , map_class=1
        , numClasses=5
        , evalMinibatches=1
        , learning_rate=.001
        , fold=0
    )

    options, args = parser.parse_args(sys.argv)
    print("Using ", options.recur_type)
    print("Current fold number: ", options.fold)
    
    (X_train, y_train), (X_test, y_test), (X_dev, y_dev), max_features, E, label_id, sequence_len = \
        aidr_hierachical.load_and_numberize_data_conv(path=options.data_dir, nb_words=options.max_features, maxlen=options.maxlen,
                                     init_type=options.init_type,
                                     dev_train_merge=1, embfile=None, 
                                     map_labels_to_five_class=options.map_class, fold=options.fold)
    
    # tf.reset_default_graph()

    # Placeholders 
    word_ids=tf.placeholder(tf.int32,shape=[None, options.maxlen, options.maxlen], name="word_id")#(i,j,k) = kth word in jth sentence in ith conversation
    sequence_lengths=tf.placeholder(tf.int32, shape=[None,options.maxlen], name="sequence_length")#(i,j) = sequence length of jth sentence in ith conversation
    number_sentences=tf.placeholder(tf.int32, shape=[None], name="number_sentences")#(i) = number of sentences in ith conversation
    sentence_id=tf.placeholder(tf.int32, shape=[None, options.maxlen],name="sentence_id" ) #(i,j) jth sentence of ith conversation
    y_values = tf.placeholder(tf.int32, [None, options.maxlen])
    labels = tf.one_hot(y_values, options.numClasses) #3dim (conv[batch_size], sentence[100], onehot representation[5])

    prediction = forward_propagation_bidirectional(word_ids, sequence_lengths, E)
    
    cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels)
    mask = tf.sequence_mask(number_sentences, options.maxlen)
    cross_ent = tf.boolean_mask(cross_ent, mask)
    loss = tf.reduce_mean(cross_ent)
    optimizer = tf.train.AdamOptimizer(learning_rate=options.learning_rate).minimize(loss)
    
    correctPred = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    y_preds = tf.argmax(prediction, axis=-1)
    
    mask = tf.reshape(tf.sequence_mask(number_sentences, options.maxlen), [-1])
    _y_values = tf.boolean_mask(tf.reshape(y_values, [-1]), mask)
    _y_preds = tf.boolean_mask(tf.reshape(y_preds, [-1]), mask)

    init = tf.global_variables_initializer()
    m = X_train.shape[0]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        saver = tf.train.Saver()
        sess.run(init)

        best_accuracy = 0.
        best_macroF1 = 0.
        best_epoch = -1
        best_minibatch = -1


        for epoch in range(options.epochs):
            # randomly shuffle the training data
            np.random.seed(2018+epoch)
            np.random.shuffle(X_train)
            np.random.seed(2018+epoch)
            np.random.shuffle(y_train)
            np.random.seed(2018+epoch)
            np.random.shuffle(sequence_len['train_seq_len'])
            np.random.seed(2018+epoch)
            np.random.shuffle(sequence_len['train_conv_len'])

            minibatch_cost = 0.
            num_minibatches = int(m / options.minibatch_size)
            train_minibatches = mini_batches(X_train, y_train, seq_len=sequence_len['train_seq_len'],
                                num_sen=sequence_len['train_conv_len'], mini_batch_size=options.minibatch_size)

            for (i, train_minibatch) in enumerate(train_minibatches):
                (train_minibatch_X, train_minibatch_y, train_minibatch_seqlen, train_minibatch_numsen) = train_minibatch
                # print("x: ", train_minibatch_X.shape, "y: ", len(train_minibatch_y), "s: ", len(train_minibatch_seqlen))
                _, train_batch_loss, pr = sess.run([optimizer, loss, prediction], {word_ids: train_minibatch_X,
                                                                                   y_values: train_minibatch_y,
                                                                                   sequence_lengths: train_minibatch_seqlen,
                                                                                   number_sentences: train_minibatch_numsen})

                # print("Iteration: ", i, "  loss: ", train_batch_loss)

                if ((i + 1) % options.evalMinibatches == 0 or i == num_minibatches - 1):
                    
                    test_y_vals, test_y_preds = sess.run([_y_values, _y_preds],
                                                        {word_ids: X_test,
                                                        y_values: y_test,
                                                        sequence_lengths:sequence_len['test_seq_len'],
                                                        number_sentences:sequence_len['test_conv_len'] })
                    
                    acc_test = metrics.accuracy_score(test_y_vals, test_y_preds)
                    # print("Test Accuracy: ", test_acc)

                    mic_p, mic_r, mic_f, sup = metrics.precision_recall_fscore_support(test_y_vals, test_y_preds,
                                                                                       average='micro')
                    mac_p, mac_r, mac_f, sup = metrics.precision_recall_fscore_support(test_y_vals, test_y_preds,
                                                                                       average='macro')

                    if (mac_f > best_macroF1):
                        best_accuracy = acc_test
                        best_macroF1 = mac_f
                        best_epoch = epoch
                        best_minibatch = i
                    
            if epoch%5 == 0 or epoch==options.epochs-1:
                print("After", epoch, " epoch.. **Best so far** Epoch: ", best_epoch, " Minibatch: ", best_minibatch,
                        " Best Test acc: ", best_accuracy, " Best F1: ", best_macroF1, " **")