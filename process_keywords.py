#coding:utf-8
import os
'''
引用OPENKG.CN里面的机器之心和清华实验室的数据集keywords。500+和16000
'''
f_r = open("./key_words/keywords8.txt",'r',encoding = 'UTF-8')#./是进入到当前目录下
f2_r = open("./key_words/THUOCL_it.txt",'r',encoding = 'UTF-8')

def read_write1(f_r):
    line = f_r.readline()
    while line:
        str = line.split("|")
        f_w = open('./key_words/keywords_list2.txt','a',encoding = 'UTF-8')
        if '/' in str[1]:
            l = str[1].split('/')
            for i in l:
                # print(i)
                f_w.write(i.strip()+'\n')
        elif '／' in str[1]:
            l = str[1].split('／')
            for i in l:
                # print(i)
                f_w.write(i.strip()+'\n')
        else:
            f_w.write(str[1].strip().replace(' ','')+'\n')
        # print(str[1])
        line = f_r.readline()
        f_w.close

def read_write2(f2_r):
    line = f2_r.readline()
    while line:
        str = line.split("	")
        f_w = open('./key_words/keywords_list2.txt','a',encoding = 'UTF-8')
        f_w.write(str[0].strip().replace(' ','')+'\n')
        print(str[0])
        line = f2_r.readline()
        f_w.close
read_write1(f_r)
# read_write2(f2_r)
f_r.close
f2_r.close