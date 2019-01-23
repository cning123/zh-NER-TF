from model import BiLSTM_CRF
import tensorflow as tf
from data import read_corpus, read_dictionary, tag2label, random_embedding
import os, argparse,time
from utils import str2bool, get_logger, get_entity


class NER_DEMO(object):
    def __init__(self,args):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        paths,model_path = get_paths(args)
        ckpt_file = tf.train.latest_checkpoint(model_path)

        paths['model_path'] = ckpt_file
        word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
        embeddings = random_embedding(word2id, args.embedding_dim)
        self.model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
        self.model.build_graph()
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=config)
        self.saver.restore(self.sess, ckpt_file)

    def  predict(self,demo_sent):
        if demo_sent == '' or demo_sent.isspace():
            print('See you next time!')
            return {}
        else:
            demo_sent = list(demo_sent.strip())
            demo_data = [(demo_sent, ['O'] * len(demo_sent))]
            tag = self.model.demo_one(self.sess, demo_data)
            entities = get_entity(tag, demo_sent)
            return entities



def get_args():
    parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
    # add_argument()方法，用来指定程序需要接受的命令参数
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
    parser.add_argument('--pretrain_embedding', type=str, default='random',
                        help='use pretrained char embedding or init it randomly')
    parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
    parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
    parser.add_argument('--demo_model', type=str, default='1547445161', help='model for test and demo')
    args = parser.parse_args()
    return args


def get_paths(args):
    paths = {}
    timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
    print(timestamp)
    output_path = os.path.join('.', args.train_data + "_save", timestamp)  # output_path:.\\data_path_save\\timestamp
    if not os.path.exists(output_path): os.makedirs(output_path)
    summary_path = os.path.join(output_path, "summaries")
    paths['summary_path'] = summary_path
    if not os.path.exists(summary_path): os.makedirs(summary_path)
    model_path = os.path.join(output_path, "checkpoints\\")
    if not os.path.exists(model_path): os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    paths['model_path'] = ckpt_prefix
    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path): os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path
    get_logger(log_path).info(str(args))

    return paths,model_path