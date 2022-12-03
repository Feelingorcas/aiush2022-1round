""" evaluation.py
Replicated in the NSML leaderboard dataset, KoreanFood.
"""

import argparse

def evaluate(prediction_labels, gt_labels, num_classes = 5 ):
    """
    Args:
      top1_reference_ids: dict(str: int)
      gt_labels: dict(str: int)
    Returns:
      acc: float top-1 accuracy.
    """
    count = 0.0
    wrong_numbers =  [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    right_numbers =  [0 for _ in range(num_classes)]
    right_numbers_rsd = [[0 for _ in range(4)] for _ in range(3)]
    for idx, query in enumerate(gt_labels):
        gt_label = int(gt_labels[query][0])
        pred_label = int(prediction_labels[query][0])

        if gt_label == pred_label:
            count += 1.0
            right_numbers[gt_label] += 1
        else :
            wrong_numbers[gt_label][pred_label] += 1
            if (len(gt_labels[query])>1) :
                right_numbers_rsd[0][int(gt_labels[query][1])] +=1
                right_numbers_rsd[1][int(gt_labels[query][2])] += 1
                right_numbers_rsd[2][int(gt_labels[query][3])] += 1



    print('wrong numbers count : {}'.format(wrong_numbers))
    print('right numbers count : {}'.format(right_numbers))
    print('wrong by race  = {}'.format(right_numbers_rsd[0]))
    print('wrong by sex  = {}'.format(right_numbers_rsd[1]))
    print('wrong by direction  = {}'.format(right_numbers_rsd[2]))
    percents = 0
    for i in range(num_classes):
        percent = right_numbers[i] / (right_numbers[i] + sum(wrong_numbers[i]))
        percents += percent
    print(percents/num_classes)

    acc = count / float(len(gt_labels))
    return acc

def read_prediction_pt(file_name):
    """
      Args:
        file_name: str
      Returns:
        top1_reference_ids: dict(str: int)
    """
    with open(file_name) as f:
        lines = f.readlines()
    dictionary = dict([l.replace('\n', '').split(' ') for l in lines])
    return dictionary

def read_prediction_gt(file_name):
    """
      Args:
        file_name: str
      Returns:
        top1_reference_ids: dict(str: int)
    """
    with open(file_name) as f:
        lines = f.readlines()
    dictionary = dict([l.replace('\n', '').split(' ') for l in lines])
    return dictionary

def evaluation_metrics(prediction_file, testset_path):

    """
      Args:
        prediction_file: str
        testset_path: str
      Returns:
        acc: float top-1 accuracy.
    """
    prediction_labels = read_prediction_pt(prediction_file)
    gt_labels = read_prediction_gt(testset_path)
    return evaluate(prediction_labels, gt_labels)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # --prediction is set by file's name that contains the result of inference. (nsml internally sets)
    # prediction file requires type casting because '\n' character can be contained.
    args.add_argument('--prediction', type=str, default='pred.txt')
    args.add_argument('--test_label_path', type=str)
    config = args.parse_args()
    # print the evaluation result
    # evaluation prints only int or float value.
    print(evaluation_metrics(config.prediction, config.test_label_path))