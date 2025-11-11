import json
import os
import pickle as pkl
import random
import argparse

import sys
sys.path.append('..')
root_dir=os.path.abspath('..')
from ref_eval import val_evid_idv, compute_image_scores

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate extra questions based on claims with a prompt. Useful for searching.')
    parser.add_argument('--eval_model', 
                        default="gemini")
    parser.add_argument('--llm_name', 
                        default="gemini-2.0-flash-001")
    parser.add_argument('--mllm_name', 
                        default="gemini-2.0-flash-001")
    parser.add_argument('--root_dir', 
                        default="")#this is the absolute path where you put AVerImaTec.
    parser.add_argument('--save_num', 
                        type=str,
                        default="4")
    parser.add_argument('--eval_type', 
                        type=str,
                        default="evidence")
    parser.add_argument('--debug', 
                        type=bool,
                        default=False)
    parser.add_argument('--seperate_val', 
                        type=bool,
                        default=False)
    parser.add_argument('--human_pred', 
                        type=bool,
                        default=False)
    parser.add_argument('--text_val', 
                        type=bool,
                        default=False)#evaluating only on the textual part of evidence
    args = parser.parse_args()

    save_str = '_'.join([args.llm_name, args.mllm_name])
    p2_data=load_json(os.path.join(args.root_dir,'data/data_clean/split_data/val.json'))
    pred_file = load_pkl(os.path.join(args.root_dir, 'fc_detailed_results', '_'.join([args.llm_name, args.mllm_name]),
                                      str(args.save_num) + '.pkl'))

    submission_res = []
    # save the results into a submission file
    for req_id in pred_file:

        results = {
            'id': req_id,
            'questions':[q['question'] for q in pred_file[req_id]['QA_info']],
            'evidence': pred_file[req_id]['evidence'],
            'verdict': pred_file[req_id]['verdict'],
            'justification': pred_file[req_id]['justification']
        }
        submission_res.append(results)
    with open('submission_' + '_'.join([args.llm_name,args.mllm_name,args.save_num]) + '.json', 'w') as json_file:
        json.dump(submission_res, json_file, indent=4)
    print('submissions/submission_' + '_'.join([args.llm_name,args.mllm_name,args.save_num]) + '.json' + ' is done...')
