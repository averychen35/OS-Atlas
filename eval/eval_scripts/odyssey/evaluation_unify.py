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

    prediction_file_path = args.prediction_file_path # for each split
    ground_truth_file_path = args.prediction_file_path # for each split

    prediction = []
    with open(prediction_file_path) as file:
        for index, line in enumerate(file):
            prediction.append(json.loads(line))

    ground_truth = []
    with open(ground_truth_file_path) as file:
        for line in file:
            ground_truth.append(json.loads(line))

    print(len(ground_truth)==len(prediction))

    # ======================================================================== #
    #                      Results on Action Match Score
    # ======================================================================== #
    num = 0
    wrong_type = 0
    click_type = 0
    scroll_type = 0
    action_match_score_dict = defaultdict(int)
    action_number_dict = defaultdict(int)
    for pred, gt in zip(prediction, ground_truth):
        try:
            print(pred["pred"].split("actions:\n")[1])
            pred_action = pred["pred"].split("actions:\n")[1].strip("<|im_end|>").strip()
        except:
            pred_action = ""
        gt_action = gt["conversations"][1]["value"].split("actions:\n")[1]
        pred_action_type = pred_action.split(" ")[0].strip(":")
        gt_action_type = gt_action.split(" ")[0]
        action_number_dict["full"] += 1
        action_number_dict[gt_action_type] += 1

        # calculate step acc based on types
        if gt_action_type==pred_action_type:
            if gt_action_type in ["CLICK","LONG_PRESS"]:  # evaluate click type
                click_type += 1
                if len(re.findall(r'\d+', pred_action))==2:
                    pred_x, pred_y = int(re.findall(r'\d+', pred_action)[0]), int(re.findall(r'\d+', pred_action)[1])
                else:
                    pred_x, pred_y = 0,0
                gt_x, gt_y = int(re.findall(r'\d+', gt_action)[0]), int(re.findall(r'\d+', gt_action)[1])
                try:
                    if math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2) <=0.14*1000:   # set 14 % of screen size as the ratio
                        # num += 1
                        action_match_score_dict["full"] += 1
                        action_match_score_dict[gt_action_type] += 1
                except:
                    continue

            elif gt_action_type=="TYPE":
                if calculate_f1_score(pred_action, gt_action)>0.5:
                    action_match_score_dict["full"] += 1
                    action_match_score_dict[gt_action_type] += 1

            elif gt_action_type=="SCROLL":
                scroll_type += 1
                if gt_action==pred_action:
                    action_match_score_dict["full"] += 1
                    action_match_score_dict[gt_action_type] += 1
            else:
                action_match_score_dict["full"] += 1
                action_match_score_dict[gt_action_type] += 1
        else:
            wrong_type += 1
    

    # Log the headers
    logger.info("="*30 + " GUI-Odyssey Results " + f"{args.split} " + "="*30)
    logger.info("Type Acc: " + str(1 - wrong_type / action_number_dict["full"]))
    logger.info("Grounding Acc: " + str(action_match_score_dict["CLICK"] / click_type))
    logger.info("Step Acc: " + str(action_match_score_dict["full"] / action_number_dict["full"]))
    logger.info("-"*20)
    logger.info("Click Type Acc: " + str(click_type / action_number_dict["CLICK"]))
    logger.info("Scroll Type Acc: " + str(scroll_type / action_number_dict["SCROLL"]))
    # Log the Action Match Score
    logger.info("Action Match Score: " + str(action_match_score_dict["full"] / action_number_dict["full"]))

    # Log the Action Match Scores for different action types
    for action_type in ["CLICK", "LONG_PRESS", "TYPE", "SCROLL", "PRESS_BACK", "PRESS_HOME", "PRESS_RECENT", "COMPLETE"]:
        try:
            logger.info(f"Action Match Score - {action_type} : {action_match_score_dict[action_type] / action_number_dict[action_type]}")
        except:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file_path', type=str, default='<prediction_file_path_each_split.jsonl>')
    parser.add_argument('--ground_truth_file_path', type=str, default='<ground_truth_file_path_each_split.jsonl>')
    parser.add_argument('--model_id', type=str, default="<model_id>")
    parser.add_argument('--split', type=str, default="random")
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--output_path', type=str, default='results/score/')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    file_handler = logging.FileHandler(os.path.join(args.output_path, "score.log"), mode='w')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    evaluate(args)