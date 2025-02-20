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

    print(len(ground_truth)==len(prediction))

    # ======================================================================== #
    #                      Results on Action Match Score
    # ======================================================================== #

    action_match_score_dict = defaultdict(int)
    action_number_dict = defaultdict(int)
    for pred, gt in zip(prediction, ground_truth):
        
        if "/web/" in gt["image"]:
            category = "web"
        else:
            category = "desktop"

        try:
            pred_action_list = pred["pred"].split("actions:\n")[1].strip("<|im_end|>").split('\n')
        except:
            pred_action_list = []
        gt_action_list = gt["conversations"][1]["value"].split("actions:\n")[1].split('\n')

        for step_idx in range(len(pred_action_list)):
            if step_idx>=len(gt_action_list):
                break

            gt_action = gt_action_list[step_idx]
            pred_action = pred_action_list[step_idx]

            pred_action_type = pred_action.split(" ")[0]
            gt_action_type = gt_action.split(" ")[0]
            action_number_dict[gt_action_type] += 1
            action_number_dict[category+"_"+gt_action_type] += 1
            action_number_dict[category] += 1
            action_number_dict["full"] += 1

            # calculate step acc based on types
            if gt_action_type==pred_action_type:
                action_match_score_dict[f"{category}_type_em"] += 1
                action_match_score_dict[gt_action_type+"_type_match_LL"] += 1
                action_match_score_dict[f"{category}_{gt_action_type}_type_em"] += 1
                if gt_action_type in ["CLICK","LONG_PRESS", "MOVETO", "RIGHTCLICK", "DOUBLECLICK"]:  # evaluate click type
                    action_match_score_dict["click_match_LL"] += 1
                    if len(re.findall(r'\d+', pred_action))==2:
                        pred_x, pred_y = int(re.findall(r'\d+', pred_action)[0]), int(re.findall(r'\d+', pred_action)[1])
                    else:
                        pred_x, pred_y = -100, -100

                    gt_x, gt_y = int(re.findall(r'\d+', gt_action)[0]), int(re.findall(r'\d+', gt_action)[1])

                    try:
                        if math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2) <=0.14*1000:  # set 14 % of screen size as the ratio
                            action_match_score_dict[category+"_"+"full"] += 1
                            action_match_score_dict[category+"_"+gt_action_type] += 1
                            action_match_score_dict[gt_action_type] += 1
                            action_match_score_dict[category] += 1
                    except:
                        continue

                elif gt_action_type == "TYPE":
                    if pred_action==gt_action or calculate_f1_score(pred_action.split()[1], gt_action.split()[1])>0.5:
                        action_match_score_dict[category+"_"+"full"] += 1
                        action_match_score_dict[category+"_"+gt_action_type] += 1
                        action_match_score_dict[category] += 1

                elif gt_action_type == "SCROLL":
                    if gt_action.split(" ")[1] in pred_action.split(" ")[1] or pred_action.split(" ")[1] in gt_action.split(" ")[1]:
                        action_match_score_dict[category+"_"+"full"] += 1
                        action_match_score_dict[category+"_"+gt_action_type] += 1
                        action_match_score_dict[category] += 1

                elif gt_action==pred_action:  # evaluate other types
                    action_match_score_dict[category+"_"+"full"] += 1
                    action_match_score_dict[category+"_"+gt_action_type] += 1
                    action_match_score_dict[category] += 1


    logger.info("="*30 + " Omniact Results " + "="*30)
    logger.info("Type EM: %f" % ((action_match_score_dict["web_type_em"] + action_match_score_dict["desktop_type_em"]) / action_number_dict["full"]))
    # logger.info("Click ACC: %f" % ((action_match_score_dict["web_CLICK"] + action_match_score_dict["desktop_CLICK"]) / (action_number_dict["web_CLICK"] + action_number_dict["desktop_CLICK"])))
    # logger.info("SCROLL ACC: %f" % ((action_match_score_dict["web_SCROLL"] + action_match_score_dict["desktop_SCROLL"]) / (action_number_dict["web_SCROLL"] + action_number_dict["desktop_SCROLL"])))

    logger.info("Step SR: %f" % ((action_match_score_dict["web_full"] + action_match_score_dict["desktop_full"]) / action_number_dict["full"]))
    logger.info("Grounding: %f" % ((action_match_score_dict["web_CLICK"] + action_match_score_dict["desktop_CLICK"]) / action_match_score_dict["CLICK_type_match_LL"]))

    logger.info("Web Type EM: %f" % (action_match_score_dict["web_type_em"] / action_number_dict["web"]))
    # logger.info("Web Click ACC: %f" % (action_match_score_dict["web_CLICK"] / action_number_dict["web_CLICK"]))
    # logger.info("Web SCROLL ACC: %f" % (action_match_score_dict["web_SCROLL"] / action_number_dict["web_SCROLL"]))
    logger.info("Web Step SR: %f" % (action_match_score_dict["web_full"] / action_number_dict["web"]))
    logger.info("Web Grounding: %f" % (action_match_score_dict["web_CLICK"] / action_match_score_dict["web_CLICK_type_em"]))


    logger.info("Desktop Type EM: %f" % (action_match_score_dict["desktop_type_em"] / action_number_dict["desktop"]))
    # logger.info("Desktop Click ACC: %f" % (action_match_score_dict["desktop_CLICK"] / action_number_dict["desktop_CLICK"]))
    logger.info("Desktop Step SR: %f" % (action_match_score_dict["desktop_full"] / action_number_dict["desktop"]))
    logger.info("Desktop Grounding: %f" % (action_match_score_dict["desktop_CLICK"] / action_match_score_dict["desktop_CLICK_type_em"]))

    # for action_type in ["CLICK", "SCROLL", "MOVETO", "HOTKEY", "DOUBLECLICK", "RIGHTCLICK","PRESS_ENTER"]:
    #     logger.info(f"Action Match Score - {action_type}: %f" % (action_match_score_dict[action_type] / action_number_dict[action_type]))
    logger.info("="*30 + " Category Results " + "="*30)
    logger.info("Web Step SR: %f" % (action_match_score_dict["web"] / action_number_dict["web"]))
    logger.info("Desktop Step SR: %f" % (action_match_score_dict["desktop"] / action_number_dict["desktop"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file_path', type=str, default='<prediction_file_path.jsonl>')
    parser.add_argument('--ground_truth_file_path', type=str, default='<ground_truth_file_path.jsonl>')
    parser.add_argument('--model_id', type=str, default="<model_id>")
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--output_path', type=str, default='results/score/')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    file_handler = logging.FileHandler(args.output_path + f"score.log", mode='w')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


    evaluate(args)