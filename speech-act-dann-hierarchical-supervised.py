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
from flip_gradient import flip_gradient
from sklearn import metrics


class dannModel(object):
    """domain adaptation model."""

    def __init__(self, E):
        self._build_model(E)

    def _build_model(self, E):
       
        # Placeholders 
        self.word_ids=tf.placeholder(tf.int32,shape=[None, options.maxlen, options.maxlen], name="word_id")#(i,j,k) = kth word in jth sentence in ith conversation
        self.sequence_lengths=tf.placeholder(tf.int32, shape=[None,options.maxlen], name="sequence_length")#(i,j) = sequence length of jth sentence in ith conversation
        self.number_sentences=tf.placeholder(tf.int32, shape=[None], name="number_sentences")#(i) = number of sentences in ith conversation
        #self.sentence_id=tf.placeholder(tf.int32, shape=[None, options.maxlen],name="sentence_id" ) #(i,j) jth sentence of ith conversation
        self.y_values = tf.placeholder(tf.int32, [None, options.maxlen])
        self.labels = tf.one_hot(self.y_values, options.numClasses) #3dim (conv[batch_size], sentence[100], onehot representation[5])


        self.domain = tf.placeholder(tf.float32, [None, options.maxlen, 2]) #one hot representation of the 2 domains: {10, 01}
        self.lambda_ = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        # RNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            # embedding matrix
            E = tf.convert_to_tensor(E, tf.float32) #2dim [num words, distriubted representation(300)]
            W_embedding = tf.get_variable("W_embedding", initializer=E)
            print("Embedding shape: ", W_embedding.shape)
            
            print("Input data shape: ", self.word_ids.shape) #3dim [conv, sentence, word id]
            data = tf.nn.embedding_lookup(W_embedding, self.word_ids)
            print("After word embedding input shape: ", data.shape) #4dim [conv, sentence, word id, distriubted representation]

            #put the time dimension (here words in sentence) on axis=1
            (_,dim_word)=np.shape(E)
            s=tf.shape(data)
            data = tf.reshape(data, [s[0] * s[1], s[2], dim_word])
            seq_len_reshaped=tf.reshape(self.sequence_lengths,[s[0]*s[1]])           

            cell_fw = tf.contrib.rnn.LSTMCell(options.hidden_size)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=options.dropout_ratio)
            cell_bw = tf.contrib.rnn.LSTMCell(options.hidden_size)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=options.dropout_ratio)

            (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, data, sequence_length=seq_len_reshaped, dtype=tf.float32)

            (c_fw, h_fw) = state_fw
            (c_bw, h_bw) = state_bw
            print("c_fw: ", c_fw.shape)
            print("c_bw: ", c_bw.shape)

            # The domain-invariant feature
            c = tf.concat([h_fw, h_bw], axis=-1) #2dim [conv * sentence, 2*hidden_size]
            #c = tf.concat([h_fw, h_bw], axis=-1)
            print("feature shape: ", c.shape)

            #put the time dimension (here sentences in conversations) on axis=1
            word_level_output = tf.reshape(c, shape=[s[0], s[1], 2*options.hidden_size])
            #word_level_output = c

            with tf.variable_scope("sentence_level"):
                cell_fw_sen = tf.contrib.rnn.LSTMCell(options.hidden_size)
                cell_fw_sen = tf.contrib.rnn.DropoutWrapper(cell=cell_fw_sen, output_keep_prob=options.dropout_ratio)
                cell_bw_sen = tf.contrib.rnn.LSTMCell(options.hidden_size)
                cell_bw_sen = tf.contrib.rnn.DropoutWrapper(cell=cell_bw_sen, output_keep_prob=options.dropout_ratio)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw_sen, cell_bw_sen, word_level_output, sequence_length=self.number_sentences, dtype=tf.float32)
                sentence_level_output = tf.concat([output_fw, output_bw], axis=-1) #2dim [conv * sentence, 2*hidden_size]
                
                print("Sentence level output shape", sentence_level_output.shape) #shape [batch_size(conv), sentences, 2*hidden_size]

            sentences = tf.shape(sentence_level_output)[1]
            sentence_level_output = tf.reshape(sentence_level_output, [-1, 2*options.hidden_size])
            self.feature = sentence_level_output #2dim [conv * sentence, 2*hidden_size]

        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = self.feature #2dim [conv * sentence, 2*hidden_size]
            source_features = tf.slice(self.feature, [0, 0], [(options.minibatch_size*options.maxlen) // 2, -1])
            target_labeled_features = tf.slice(self.feature, [options.minibatch_size*options.maxlen // 2, 0], [options.minibatch_size*options.maxlen // 2, -1])
            print("All features: ", all_features.shape)
            print("source features: ", source_features.shape)
            print("target labeled features: ", target_labeled_features.shape)
            classify_feats = tf.cond(self.train, lambda: tf.concat([source_features, target_labeled_features], 0), lambda: all_features)
            
            all_labels = self.labels #3dim (conv, sentence, onehot representation[5])
            source_labels = tf.slice(self.labels, [0, 0, 0], [(options.minibatch_size) // 2, -1, -1])
            target_labeled_labels = tf.slice(self.labels, [options.minibatch_size // 2, 0, 0], [options.minibatch_size // 2, -1, -1])
            print("All labels: ", all_labels.shape)
            print("source labels: ", source_labels.shape)
            print("target labelled labels: ", target_labeled_labels.shape)
            self.classify_labels = tf.cond(self.train, lambda: tf.concat([source_labels, target_labeled_labels], 0), lambda: all_labels)
            
            all_number_sentences = self.number_sentences #1D [conv]
            source_number_sentences = tf.slice(self.number_sentences, [0], [(options.minibatch_size) // 2])
            target_number_sentences = tf.slice(self.number_sentences, [(options.minibatch_size) // 2], [(options.minibatch_size) // 2])
            print("All number of sentences: ", all_number_sentences.shape)
            print("source number of sentences: ", source_number_sentences.shape)
            print("target number of sentences: ", target_number_sentences.shape)
            self.classify_number_sentences = tf.cond(self.train, lambda: tf.concat([source_number_sentences, target_number_sentences], 0), lambda: all_number_sentences)
           
            weight = tf.get_variable("l_w1",
                                     shape=[2*options.hidden_size, options.numClasses],
                                     initializer=tf.contrib.layers.xavier_initializer(seed=101))
            bias = tf.get_variable("l_b1", shape=[options.numClasses], initializer=tf.constant_initializer(0.0))

            logits = (tf.matmul(classify_feats, weight) + bias) #2dim [conv * sentence, numClasses]
            self.logits = tf.reshape(logits, [-1, sentences, options.numClasses])
            print("label Predictor: ", self.logits.shape) #3dim [conv, sentence, numClasses]


            self.pred = tf.nn.softmax(self.logits)
            mask_label_pred = tf.reshape(tf.sequence_mask(self.classify_number_sentences, options.maxlen), [-1]) #1 dim [conv, maxlen(100)]
            masked_logits = tf.boolean_mask(tf.reshape(self.logits, [-1, options.numClasses]), mask_label_pred)   
            masked_labels = tf.boolean_mask(tf.reshape(self.classify_labels, [-1, options.numClasses]), mask_label_pred)
            print("label mask: ", mask_label_pred.shape)
            print("label masked_logits: ", masked_logits.shape)
            print("label masked_labels: ", masked_labels.shape)

            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=masked_logits, labels=masked_labels)
            print("pred loss: ", self.pred_loss.shape)


        # MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.lambda_) #2dim [conv * sentence, 2*hidden_size]

            d_W_fc0 = tf.get_variable("d_w1", shape=[2*options.hidden_size, 100],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=101))
            d_b_fc0 = tf.get_variable("d_b1", shape=[100], initializer=tf.constant_initializer(0.0))
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0) #2dim [conv * sentence,100]

            d_W_fc1 = tf.get_variable("d_w2", shape=[100, 2],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=101))
            d_b_fc1 = tf.get_variable("d_b2", shape=[2], initializer=tf.constant_initializer(0.0))
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1 #2dim [conv * sentence, 2]
            print("domain predictor: ", d_logits.shape)
            
            mask_domain_pred = tf.reshape(tf.sequence_mask(self.number_sentences, options.maxlen), [-1])
            masked_d_logits = tf.boolean_mask(d_logits, mask_domain_pred)
            masked_domain = tf.boolean_mask(tf.reshape(self.domain, [-1, 2]), mask_domain_pred)
            print("domain mask: ", mask_domain_pred.shape)
            print("domain masked_logits: ", masked_d_logits.shape)
            print("domain masked_labels: ", masked_domain.shape)

            #self.domain_pred = tf.nn.softmax(d_logits_masked)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=masked_d_logits, labels=masked_domain)
            
            print("domain loss: ", self.domain_loss.shape)


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
        data_dir="./data/new_data/New Speech-act Data/ta/"
        , data_spec="in"

        , model_dir="./saved_models/"
        , log_file="log"

        , learn_alg="momentum"  # momentum, sgd, adagrad, rmsprop, adadelta, adam (default)
        , loss="softmax_crossentropy"  # hinge, squared_hinge, binary_crossentropy (default)
        , minibatch_size=10
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
        , map_class=0
        , numClasses=5
        , evalMinibatches=5
        , fold=0
    )

    options, args = parser.parse_args(sys.argv)
    print("Using ", options.learn_alg, "optimizer.")
    print("Current fold number: ", options.fold)

    # path = "../../Documents/Projects/Speech_Act/DA_tagger/data/input_to_DNNs/cat_MRDA/QL"
    (X_src, y_src), (X_train, y_train), (X_test, y_test), (X_dev, y_dev), max_features, E, label_id, sequence_len = \
        aidr_hierachical.load_and_numberize_data_conv_dann(path=options.data_dir, nb_words=options.max_features, maxlen=options.maxlen,
                                     init_type=options.init_type,
                                     dev_train_merge=1, embfile=None, 
                                     map_labels_to_five_class=1, fold=options.fold)

    model = dannModel(E)

    learning_rate = tf.placeholder(tf.float32, [])

    prediction = model.logits #3dim [conv, sentence, numClasses]

    pred_loss = tf.reduce_mean(model.pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss

    if options.learn_alg == "adam":
        optimizer_regular = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(pred_loss)
        optimizer_dann = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
    elif options.learn_alg == "momentum":
        optimizer_regular = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
        optimizer_dann = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)
    
    y_preds = tf.argmax(prediction, axis=-1) #2dim [conv, sentence]
    mask = tf.reshape(tf.sequence_mask(model.number_sentences, options.maxlen), [-1])
    _y_values = tf.boolean_mask(tf.reshape(model.y_values, [-1]), mask)
    _y_preds = tf.boolean_mask(tf.reshape(y_preds, [-1]), mask)

    init = tf.global_variables_initializer()
    m = X_src.shape[0]

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
            # randomly shuffle the source data
            np.random.seed(2018+epoch)
            np.random.shuffle(X_src)
            np.random.seed(2018+epoch)
            np.random.shuffle(y_src)
            np.random.seed(2018+epoch)
            np.random.shuffle(sequence_len['src_seq_len'])
            np.random.seed(2018+epoch)
            np.random.shuffle(sequence_len['src_conv_len'])

            src_minibatches = mini_batches(X_src, y_src, seq_len=sequence_len['src_seq_len'], num_sen=sequence_len['src_conv_len'],
                                           mini_batch_size=options.minibatch_size // 2)
            src_num_minibatches = len(src_minibatches)
            #print("Total source minibatches: ", src_num_minibatches)

            # randomly shuffle the target training data
            np.random.seed(2018+epoch)
            np.random.shuffle(X_train)
            np.random.seed(2018+epoch)
            np.random.shuffle(y_train)
            np.random.seed(2018+epoch)
            np.random.shuffle(sequence_len['train_seq_len'])
            np.random.seed(2018+epoch)
            np.random.shuffle(sequence_len['train_conv_len'])

            target_train_minibatches = mini_batches(X_train, y_train, seq_len=sequence_len['train_seq_len'], num_sen=sequence_len['train_conv_len'],
                                                    mini_batch_size=options.minibatch_size // 2)
            target_num_minibatches = len(target_train_minibatches)

            domain_labels = np.vstack([np.tile([1., 0.], [options.minibatch_size // 2, options.maxlen, 1]),
                                       np.tile([0., 1.], [options.minibatch_size // 2, options.maxlen, 1])])

            for (i, src_minibatch) in enumerate(src_minibatches):

                # Adaptation param and learning rate schedule as described in the paper
                num_steps = src_num_minibatches*options.epochs
                p = float(epoch*src_num_minibatches + i) / num_steps
                lambda_ = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.5 / (1. + 2 * p) ** 0.75

                (src_minibatch_X, src_minibatch_y, src_minibatch_seqlen, src_minibatch_convlen) = src_minibatch
                (target_train_minibatch_X, target_train_minibatch_y, target_train_minibatch_seqlen, target_train_minibatch_convlen) = \
                target_train_minibatches[i%target_num_minibatches]
                
                X = np.vstack((src_minibatch_X, target_train_minibatch_X))
                Y = np.vstack((src_minibatch_y, target_train_minibatch_y))
                Z_seq_len=np.vstack((src_minibatch_seqlen, target_train_minibatch_seqlen))
                Z_num_sen=np.hstack((src_minibatch_convlen, target_train_minibatch_convlen))
                
                _, batch_loss = sess.run([optimizer_dann, total_loss],
                                         feed_dict={model.word_ids: X, model.y_values: Y, 
                                                    model.sequence_lengths: Z_seq_len, model.number_sentences: Z_num_sen,
                                                    model.domain: domain_labels,
                                                    model.train: True, model.lambda_: lambda_, learning_rate: lr})


                if ((i + 1) % options.evalMinibatches == 0 or i == len(src_minibatches) - 1):

                    test_y_vals, test_y_preds = sess.run([_y_values, _y_preds],
                    feed_dict={model.word_ids: X_test, model.y_values: y_test,
                               model.sequence_lengths: sequence_len['test_seq_len'], 
                               model.number_sentences: sequence_len['test_conv_len'],
                               model.train: False})

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
                        conf_matrix = metrics.confusion_matrix(test_y_vals, test_y_preds)

            if epoch%5 == 0 or epoch==options.epochs-1:
                print("After", epoch, " epoch.. **Best so far** Epoch: ", best_epoch, " Minibatch: ", best_minibatch,
                        " Best Test acc: ", best_accuracy, " Best F1: ", best_macroF1, " **")
        print("\nConfusion Matrix:\n", conf_matrix)
