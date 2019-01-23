import pickle
f = open('./data_path/word2id.pkl','rb+')
data = pickle.loads(f.read())
print(data)
