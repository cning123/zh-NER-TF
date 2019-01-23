# -*- coding: utf-8 -*-
import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# def get_entity(tag_seq, char_seq):
#     PER = get_PER_entity(tag_seq, char_seq)
#     LOC = get_LOC_entity(tag_seq, char_seq)
#     ORG = get_ORG_entity(tag_seq, char_seq)
#     TIME = get_TIME_entity(tag_seq, char_seq)
#     ROLE = get_ROLE_entity(tag_seq, char_seq)
#     return PER, LOC, ORG, TIME, ROLE

def get_entity(tag_seq,char_seq):
    # ALG,MDL,TECH,OPQ,CHAR = [],[],[],[],[]
    entities = {}
    entity = ''
    tag = ''
    for index,i in enumerate(tag_seq):
        if '-' in str(i):
            tag = i.split('-')[-1]
            entity+=char_seq[index]

        is_end = (index == len(tag_seq)-1)
        if ((is_end and tag_seq[index-1] != 0) or (not is_end and tag_seq[index+1] == 0 and tag_seq[index-1] != 0)) and i != 0:
            tag_entities = entities[tag] if tag in entities.keys() else []
            tag_entities.append(entity)
            entities[tag] = tag_entities
            entity = ''


    return entities

# def get_ALG_entity(tag_seq,char_seq):#算法
#     length = len(char_seq)
#     ALG = []
#     for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
#         if tag == 'B-ALG':
#             pass
    #         if 'alg' in locals().keys():
    #             ALG.append(alg)
    #             del alg
    #         per = char
    #         if i + 1 == length:
    #             PER.append(per)
    #     if tag == 'I-PER':
    #         per += char
    #         if i + 1 == length:
    #             PER.append(per)
    #     if tag not in ['I-PER', 'B-PER']:
    #         if 'per' in locals().keys():
    #             PER.append(per)
    #             del per
    #         continue
    # return PER


def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-PER':
            per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i+1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i+1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG

def get_TIME_entity(tag_seq, char_seq):
    length = len(char_seq)
    TIME = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-TIME':
            if 'time' in locals().keys():
                TIME.append(time)
                del time
            time = char
            if i+1 == length:
                TIME.append(time)
        if tag == 'I-TIME':
            time += char
            if i+1 == length:
                TIME.append(time)
        if tag not in ['I-TIME', 'B-TIME']:
            if 'time' in locals().keys():
                TIME.append(time)
                del time
            continue
    return TIME

def get_ROLE_entity(tag_seq, char_seq):
    length = len(char_seq)
    ROLE = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ROLE':
            if 'role' in locals().keys():
                ROLE.append(role)
                del role
            role = char
            if i+1 == length:
                ROLE.append(role)
        if tag == 'I-ROLE':
            role += char
            if i+1 == length:
                ROLE.append(role)
        if tag not in ['I-ROLE', 'B-ROLE']:
            if 'role' in locals().keys():
                ROLE.append(role)
                del role
            continue
    return ROLE


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
