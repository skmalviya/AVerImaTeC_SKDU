import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

import json
import pickle as pkl
from scipy import stats


def load_pkl(path):
    data = pkl.load(open(path, 'rb'))
    return data


def load_json(path):
    data = json.load(open(path, 'r'))
    return data


import pandas as pd
from collections import defaultdict

import os
import random

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

import nltk
from nltk import word_tokenize


def pairwise_meteor(candidate, reference):  # Todo this is not thread safe, no idea how to make it so
    return nltk.translate.meteor_score.single_meteor_score(word_tokenize(reference), word_tokenize(candidate))


path = os.path.abspath('.')
p2_data = load_json(os.path.join(path, 'data/data_clean/split_data/val.json'))
test_id_to_row = {}
for i, row in enumerate(p2_data):
    test_id_to_row[i] = row


def compute_scores_detail(score, len_gt_evid, len_val_evid):
    if len_val_evid == 0 or len_gt_evid == 0:
        return None, None, None
    # print (score["pred_in_ref"])
    precision = score["pred_in_ref"] / len_val_evid
    recall = score["ref_in_pred"] / len_gt_evid
    if precision < 0:
        precision = 0
    if recall < 0:
        recall = 0
    if recall > 1.:
        recall = 1.
    if precision > 1.:
        precision = 1.
    if precision == 0 and recall == 0:
        f1 = 0.
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def get_auto_recall(scores, req_id):
    result = scores[req_id]
    ref_evid = result['ref_evid']
    pred_evid = result['pred_evid']
    # print (result)
    pred_in_ref = result['image_scores']['pred_in_ref']
    ref_in_pred = result['image_scores']['ref_in_pred']
    # print (pred_in_ref)
    pred_dict = defaultdict(int)
    len_pred = len(pred_evid)
    # print (ref_in_pred)
    num_pred_in_ref = 0
    for i, info in enumerate(pred_in_ref):
        # print (info)
        try:
            pred_idx = int(info['info'][0].split('_')[-1])
            ref_idx = int(info['info'][1].split('_')[-1])
        except:
            continue
        if pred_idx in pred_dict:
            continue
        pred_dict[pred_idx] += 1
        try:
            if int(info['score']) < threshold:
                continue
            else:
                num_pred_in_ref += 1
        except:
            continue
    ref_dict = defaultdict(int)
    num_ref_in_pred = 0
    for i, info in enumerate(ref_in_pred):
        # print (info)
        try:
            pred_idx = int(info['info'][1].split('_')[-1])
            ref_idx = int(info['info'][0].split('_')[-1])
        except:
            continue
        if ref_idx in ref_dict:
            continue
        ref_dict[ref_idx] += 1
        # print (ref_dict)
        try:
            if int(info['score']) < threshold:
                continue
            else:
                num_ref_in_pred += 1
        except:
            continue
    """
    print (result['detailed_val'])
    print (len(pred_evid),len(ref_evid))
    print ('\t',pred_in_ref)
    print ('\t',ref_in_pred)
    print ('\t',num_pred_in_ref,num_ref_in_pred)
    """
    precision, recall, f1 = compute_scores_detail({'ref_in_pred': num_ref_in_pred, 'pred_in_ref': num_pred_in_ref},
                                                  len(ref_evid), len_pred)
    return precision, recall, f1


llm_name = 'llama'
mllm_name = 'llava'

llm_name = 'qwen'
mllm_name = 'qwen'

# llm_name='qwen'
# mllm_name='llava'

# llm_name='gemma'
# mllm_name='gemma'

# llm_name = "gemini-2.0-flash-001"
# mllm_name = "gemini-2.0-flash-001"

# llm_name="o3-2025-04-16"
# mllm_name="o3-2025-04-16"
save_num = '17'
threshold = 9

pred_file = load_pkl(os.path.join(path, 'fc_detailed_results', '_'.join([llm_name, mllm_name]), save_num + '.pkl'))
print(len(pred_file))

avg = 0
for req_id in pred_file:
    pred_questions = pred_file[req_id]['QA_info']
    avg += len(pred_questions)
print(avg * 1. / len(pred_file))

scores = load_pkl(os.path.join(path,
                               "evaluation",
                               'intermediate_info/' + '_'.join([llm_name, mllm_name]) + '_val_evid_' + str(
                                   save_num) + '_raw.pkl'))
print(len(scores))

justifications = load_pkl(
    os.path.join(path, 'fc_detailed_results', '_'.join([llm_name, mllm_name]), save_num + '_justification.pkl'))

prec = 0.
rec = 0.
f1s = 0.
valid = 0
gen_questions = 0
para_ques = (save_num in ['4', '5', '14', '13', '11'])
for req_id in scores:
    precision, recall, f1 = get_auto_recall(scores, req_id)
    # print (recall)
    if precision is None or recall is None:
        continue
    valid += 1

    prec += precision
    rec += recall
    f1s += f1
print(valid, gen_questions / valid)
print('Prec:', prec * 100. / valid)
print('Rec:', rec * 100. / valid)
print(f1s * 100. / valid)

acc = 0.
valid = 0
meteor = 0.
tr = 0.4
for req_id in scores:
    precision, recall, f1 = get_auto_recall(scores, req_id)

    if precision is None or recall is None:
        continue
    valid += 1
    gt_label = test_id_to_row[req_id]['label']
    pred_label = pred_file[req_id]['verdict']
    # pred_just=' '.join(pred_file[req_id]['justification'].split('\n'))
    pred_just = justifications[req_id]
    # pred_just=pred_file[req_id]['justification']
    gt_just = test_id_to_row[req_id]['justification']
    print('[PRED]:', pred_just)
    print('[REF]:', gt_just)
    if gt_label == pred_label and recall > tr:
        acc += 1
    if recall > tr:
        # mt=pairwise_meteor(pred_just, gt_just)
        mt = scorer.score(gt_just, pred_just)['rouge1'].recall
        # print (pred_just)
        meteor += mt
        # print (mt)
print(valid)
print(acc / valid)
print(meteor / valid)

