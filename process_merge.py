# -*- coding: UTF-8 -*-
import random
#分割数据为train和test
def segmentation():
    f = open('./data_path/merge.csv', 'r', encoding='utf-8')
    key = '13092'
    alist = []
    adict = {}
    for lines in f.readlines():
        list = lines.split(",")
        blist = []
        blist.append(list[1])
        blist.append(list[2])
        if key == list[0]:
            alist.append(blist)
        else:
            #说明到下一句话了
            adict[key] = alist
            alist = []
            alist.append(blist)
            key = list[0]
    return adict

def seg_class(adict):
    alg_list = []
    mdl_list = []
    tech_list = []
    opq_list = []
    char_list = []
    for key in adict:
        for list in adict[key]:#['大'，‘O\n]
            # list = i.split(",")
            if '算法' in list[1]:
                if key not in alg_list:
                    alg_list.append(key)
                # break
            if '模型' in list[1]:
                if key not in mdl_list:
                    mdl_list.append(key)
                # break
            if '技术' in list[1]:
                if key not in tech_list:
                    tech_list.append(key)
                # break
            if '未决问题' in list[1]:
                if key not in opq_list:
                    opq_list.append(key)
                # break
            if '特性' in list[1]:
                if key not in char_list:
                    char_list.append(key)
                # break
    print(len(alg_list),len(mdl_list))
    return alg_list,mdl_list,tech_list,opq_list,char_list

def randomshuffle(adict,list):
    train_write = open('./data_path/train_merge.txt', 'a', encoding='utf-8')
    test_write = open('./data_path/test_merge.txt', 'a', encoding='utf-8')
    len_test = len(list)//10
    len_train = len(list) - len_test
    for i in list[:len_train]:
        for j in adict[i]:#['char','tag']
            train_write.write(i+' '+j[0]+' '+j[1])
            # for l in j:
            #     train_write.write(i+","+l+',')
            # train_write.write("".join(j[0]).join(j[1])+'\n')
    for i in list[len_train: ]:
        for j in adict[i]:
            test_write.write(i + ' ' + j[0] + ' ' + j[1])
            # for l in j:
            #     test_write.write(i+","+l+',')
            # test_write.write("".join(j[0]).join(j[1])+'\n')


adict = segmentation()
alg_list,mdl_list,tech_list,opq_list,char_list = seg_class(adict)
random.shuffle(alg_list)
random.shuffle(mdl_list)
random.shuffle(tech_list)
random.shuffle(opq_list)
random.shuffle(char_list)
# train_write = open('./data_path/train_merge.txt','w+',encoding='utf-8')
# test_write = open('./data_path/test_merge.txt','w+',encoding='utf-8')
randomshuffle(adict,alg_list)
randomshuffle(adict,mdl_list)
randomshuffle(adict,tech_list)
randomshuffle(adict,opq_list)
randomshuffle(adict,char_list)
