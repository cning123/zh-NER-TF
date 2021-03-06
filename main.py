import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding


## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory


## hyperparameters
#创建解析器对象ArgumentParser，可以添加参数
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
#add_argument()方法，用来指定程序需要接受的命令参数
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1547445161', help='model for test and demo')
args = parser.parse_args()

#1551864803是新数据model 2019-03-06 downloadfile3-4
#1552104107是train_data训练的结果2019-0309
#1552660437是train_merge和test_merge训练测试出来的

## get char embeddings
word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)#(3905,300)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


## read corpus and get training data
if args.mode != 'train':
    # train_path = os.path.join('.', args.train_data, 'train_data')
    # test_path = os.path.join('.', args.test_data, 'test_data')
    train_path = os.path.join('.', args.train_data, 'processed_downloadfile3')
    test_path = os.path.join('.', args.test_data, 'processed_downloadfile4')
    train_data = read_corpus(train_path)#list[（句子，label），（句子，label）]
    test_data = read_corpus(test_path)
    test_size = len(test_data)#test中有多少条句子


## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
print(timestamp)
output_path = os.path.join('.', args.train_data+"_save", timestamp)#output_path:.\\data_path_save\\timestamp
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")#summary_path:./data_path_save/timestamp/summaries
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints\\")#model_path:.\\data_path_save\\timestamp\\checkpoints
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")#ckpt_prefix:.\\data_path_save\\timestamp\\model
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")#result_path:.\\data_path_save\\timestamp\\results
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")#log_path:.\\data_path_save\\timestamp\\results\\log.txt
paths['log_path'] = log_path
get_logger(log_path).info(str(args))#记录日志


## training model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    train_path = os.path.join('.', args.train_data, 'processed_downloadfile3')################
    # train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'processed_downloadfile4')
    # test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    # test_data = train_data[:300]#############

    # hyperparameters-tuning, split train/dev
    dev_data = train_data[:3000]; dev_size = len(dev_data)
    train_data = train_data[3000:]; train_size = len(train_data)
    print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    #50658
    print('==========================================')
    print('==========================================')
    print('==========================================')
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        #使用 saver.restore() 方法，重载模型的参数，继续训练或用于测试数据。
        saver.restore(sess, ckpt_file)
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                entities = get_entity(tag, demo_sent)
                print({i:entities[i] for i in entities.keys()})
                #print('PER: {}\nLOC: {}\nORG: {}\nTIME: {}\nROLE: {}'.format(PER, LOC, ORG, TIME, ROLE))
