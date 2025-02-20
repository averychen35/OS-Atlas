import numpy as np
import json
from collections import defaultdict
import argparse
import re
import math
import os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def evaluate(args):
    prediction_file_path = args.prediction_file_path
    ground_truth_file_path = args.ground_truth_file_path
    prediction = []
    with open(prediction_file_path) as file:
        for line in file:
            prediction.append(json.loads(line))

    ground_truth = []
    with open(ground_truth_file_path) as file:
        for line in file:
            ground_truth.append(json.loads(line))

    # processed subsplits file to category
    with open(f"android_control_test_subsplits.json","r") as file:
        test_subsplits = json.load(file)

    # ======================================================================== #
    #                          Results on Low-level
    # ======================================================================== #
    mis_click_wait_num = 0
    step_acc_res_dict = defaultdict(int)
    sample_number_dict = defaultdict(int)
    for pred, gt in zip(prediction, ground_truth):

        try:
            pred_action = pred["pred"].split("actions:\n")[1].strip("<|im_end|>")   # <im_end> for qwen2-vl
        except:
            pred_action = "invalid action"

        gt_action = gt["conversations"][1]["value"].split("actions:\n")[1]
        episode_id = int(gt["image"].split("/")[-1].split("_")[1]) # parse out the episode index
        subsplit_type = next((category for category, ids in test_subsplits.items() if episode_id in ids), None)
        sample_number_dict[subsplit_type] += 1
        sample_number_dict["full"] += 1

        sample_number_dict[gt_action.split()[0]] += 1

        gt_action_type = gt_action.split()[0]
        pred_action_type = pred_action.split()[0]

        # calculate step acc based on types
        if gt_action_type==pred_action_type:
            step_acc_res_dict["type_match"] += 1
            step_acc_res_dict[gt_action_type+"_type_match"] += 1
            if gt_action_type in ["CLICK","LONG_PRESS"]:  # evaluate click type
                step_acc_res_dict["click_match"] += 1
                try:
                    pred_x, pred_y = int(re.findall(r'\d+', pred_action)[0]), int(re.findall(r'\d+', pred_action)[1])
                except:
                    pred_x, pred_y = -100, -100
                gt_x, gt_y = int(re.findall(r'\d+', gt_action)[0]), int(re.findall(r'\d+', gt_action)[1])

                if math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2) <=0.14*1000:  # set 14 % of screen size as the ratio
                    step_acc_res_dict[subsplit_type] += 1
                    step_acc_res_dict["full"] += 1
                    step_acc_res_dict[gt_action_type+"_all_match"] += 1

            elif gt_action_type == "OPEN_APP":
                if gt_action==pred_action or calculate_f1_score(pred_action.split()[1], gt_action.split()[1])>0.5:
                    step_acc_res_dict[subsplit_type] += 1
                    step_acc_res_dict["full"] += 1
                    step_acc_res_dict[gt_action_type+"_all_match"] += 1

            elif gt_action_type == "TYPE":
                if pred_action==gt_action or calculate_f1_score(pred_action.split()[1], gt_action.split()[1])>0.5:
                    step_acc_res_dict[subsplit_type] += 1
                    step_acc_res_dict["full"] += 1
                    step_acc_res_dict[gt_action_type+"_all_match"] += 1

            elif gt_action==pred_action:  # evaluate other types
                step_acc_res_dict[subsplit_type] += 1
                step_acc_res_dict["full"] += 1
                step_acc_res_dict[gt_action_type+"_all_match"] += 1


    # Print the low-level results
    logger.info("="*30 + " AC Step Acc " + f"{args.split} " + "="*30)
    logger.info("Acc: %f" % (step_acc_res_dict["full"] / sample_number_dict["full"]))
    logger.info("iid: %f" % (step_acc_res_dict["IDD"] / sample_number_dict["IDD"]))
    logger.info("app_unseen: %f" % (step_acc_res_dict["app_unseen"] / sample_number_dict["app_unseen"]))
    logger.info("task_unseen: %f" % (step_acc_res_dict["task_unseen"] / sample_number_dict["task_unseen"]))
    logger.info("category_unseen: %f" % (step_acc_res_dict["category_unseen"] / sample_number_dict["category_unseen"]))

    logger.info(f"type_match acc: %f" % (step_acc_res_dict[f"type_match"] / sample_number_dict[f"full"]))
    logger.info(f"grounding acc: %f" % (step_acc_res_dict[f"CLICK_all_match"] / step_acc_res_dict[f"CLICK_type_match"]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file_path', type=str, default='<prediction_file.jsonl>')
    parser.add_argument('--ground_truth_file_path', type=str, default='<ground_truth_file.jsonl>')
    parser.add_argument('--model_id', type=str, default="")
    parser.add_argument('--split', type=str, default="low")
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--output_path', type=str, default='results/score/')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    file_handler = logging.FileHandler(args.output_path + f"score.log", mode="w")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    evaluate(args)
