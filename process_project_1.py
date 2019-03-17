# -*- coding: UTF-8 -*-

def pro_file():
    f = open('./data_path/test_merge.txt','r',encoding='utf-8')
    f_write = open('./data_path/processed_test_merge.txt','w+',encoding='utf-8')#download3 is train data;download4 is test data
    i=0
    num0=0
    for line in f.readlines():
        i+=1
        list_words = line.split(' ')
        if len(list_words)> 3:
            alist = []
            alist.append(list_words[0])
            alist.append(',')
            alist.append('O\n')
            list_words = alist
        print(list_words)
        if(i==1):
            num1 = int(list_words[0])
        num2 = int(list_words[0])
        res = num2-num1
        print(res)
        # if(num2-num1>=1 and num2-num1<=50 ):
        if num2 != num1:
            f_write.write(''+'\n')
            num1=num2
        if(list_words[1]==' '):
            continue
        tag = trans(list_words[-1])
        if(tag=='out'):
            continue
        if(len(list_words)==3) :
            f_write.write(list_words[1]+' ')
            f_write.write(tag+'\n')

def trans(list_words):
    #中文tag换成英文tag
    if(list_words=='O\n'):
        return 'O'
    else:
        tag_c = list_words.split('-')
        if(tag_c[1]=='算法\n'):
            return tag_c[0]+'-'+'ALG'
        elif(tag_c[1]=='模型\n'):
            return tag_c[0]+'-'+'MDL'
        elif(tag_c[1]=='技术\n'):
            return tag_c[0]+'-'+'TECH'
        elif(tag_c[1]==('未决问题\n')):
            return tag_c[0]+'-'+'OPQ'
        elif(tag_c[1]=='特性\n'):
            return tag_c[0]+'-'+'CHAR'
        else:
            return 'out'


pro_file()
