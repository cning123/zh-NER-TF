import sys, pickle, os, random
import numpy as np

## tags, BIO
# tag2label = {"O": 0,
#              "B-PER": 1, "I-PER": 2,
#              "B-LOC": 3, "I-LOC": 4,
#              "B-ORG": 5, "I-ORG": 6,
#              "B-TIME":7, "I-TIME":8,
#              "B-ROLE":9, "I-ROLE":10,
#              "B-CRIME":11, "I-CRIME":12,
#              "B-LAW":13, "I-LAW":14
#              }
# tag2label = {"O": 0,
#              "B-PER": 1, "I-PER": 2,
#              "B-LOC": 3, "I-LOC": 4,
#              "B-ORG": 5, "I-ORG": 6
#              }
tag2label = {"O":0,
             "B-ALG":1,"I-ALG":2,
             "B-MDL":3,"I-MDL":4,
             "B-TECH":5,"I-TECH":6,
             "B-OPQ":7,"I-OPQ":8,
             "B-CHAR":9,"I-CHAR":10,
             "B_TECH":11}

def read_corpus(corpus_path):#读data，返回char label组成的data
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data，list形式。文件下的所有句子以及对应的label，以（句子，label）（句子，label）形式的list返回
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            try:
                [char, label] = line.strip().split(' ')#字，B-LABEL
                sent_.append(char)#句子中所有的字
                tag_.append(label)#句子中所有的label
            except Exception as e:
                print(e)
                pass
        else:
            data.append((sent_, tag_))#data：句子和label两个list组成的tuple加入到datalist中
            sent_, tag_ = [], []
    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

#将句子里的每个字返回其对应在字典里的下标值
#eg:['于', '大', '宝', '的', '进', '球', '帮', '助', '中', '国', '队', '在', '长', '沙', '贺', '龙', '体', '育', '中', '心', '以', '1', '-', '0', '的', '比', '分', '获', '胜']
# [273, 55, 1071, 8, 430, 1912, 1092, 7, 52, 21, 569, 73, 14, 2065, 2405, 600, 922, 451, 52, 237, 134, 94, 3904, 94, 8, 805, 786, 725, 831]
def sentence2id(sent, word2id):
    """

    :param sent:--句子
    :param word2id:--字典,每个字对应一个数字
    :return:sentence_id--一句话中每个字对应的字典下标
    """
    sentence_id = []
    for word in sent:
        #word是数字的情况，贴NUM标签
        if word.isdigit():
            word = '<NUM>'
        #word是a-z或者A-Z的情况，贴ENG标签
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        #word不在字典里
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
        #返回字典里的下标
    return sentence_id

#返回一个word2id的dict，长度为3905，eg：{‘字1’：2201，‘字2’：599...}
def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))#list_size:3905
    return word2id

#生成一个随机的（3905，300）维array

def random_embedding(vocab, embedding_dim):
    """
    :param vocab:
    :param embedding_dim:
    :return:
    """
    #从均匀分布中抽取样本。样品在半开区间均匀分布 （包括低，但不包括高）。
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)#shape(3905,300)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:扩充或截断的句子序列以及min(len(seq), max_len)
    """
    #29
    #获取sequences中所有list的最长长度
    max_len = max(map(lambda x : len(x), sequences))#135
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)#sequences中扩充好的list
        seq_len_list.append(min(len(seq), max_len))#句子长度和最大长度的min
    return seq_list, seq_len_list

#返回字典下标seqs，以及标签labels
#eg：seqs[[273, 55, 1071, 8, 430, 1912, 1092, 7, 52, 21, 569, 73, 14, 2065, 2405, 600, 922, 451, 52, 237, 134, 94, 3904, 94, 8, 805, 786, 725, 831]]
#labels [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:--traindata
    :param batch_size:default64
    :param vocab:--word2id
    :param tag2label:--tag2label
    :param shuffle:
    :return:seqs--batch_size句子中每句话每个字对应的字典下标
            labels--batch_size句子中每句话每个字的标签
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent, tag_) in data:
        sent_ = sentence2id(sent, vocab)#返回句子中每个字在字典中的下标
        label_ = [tag2label[tag] for tag in tag_]#一句话中每个字的标签

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

