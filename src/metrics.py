""" 
metrics utilized for evaluating multi-label classification system
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

import icecream as ic


def precision(
    output: torch.Tensor, target: torch.Tensor, classe: int = 1, epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Calculate precision metric for a classification task.
    """
    _, predicted = torch.max(output, 1)

    assert predicted.shape == target.shape

    true_positive = torch.sum((predicted == target) * (target == classe))
    false_positive = torch.sum((predicted != target) * (predicted == classe))
    precision = true_positive / (true_positive + false_positive + epsilon)

    return precision, true_positive, false_positive


import torch


def recall(
    output: torch.Tensor, target: torch.Tensor, classe: int = 1, epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Calculate the recall metric for a classification task.
    """
    _, predicted = torch.max(output, 1)

    assert predicted.shape == target.shape

    true_positive = torch.sum((predicted == target) * (target == classe))
    false_negative = torch.sum((predicted != target) * (target == classe))
    recall = true_positive / (true_positive + false_negative + epsilon)

    return recall, true_positive, false_negative


def f1_score(
    output: torch.Tensor,
    target: torch.Tensor,
    classe: torch.Tensor = torch.tensor(1),
    epsilon: torch.Tensor = torch.tensor(1e-7),
) -> torch.Tensor:
    """
    Calculate the F1 score for a classification task.
    """
    _, predicted = torch.max(output, 1)

    assert predicted.shape == target.shape

    true_positive = torch.sum((predicted == target) * (target == classe))
    false_positive = torch.sum((predicted != target) * (predicted == classe))
    false_negative = torch.sum((predicted != target) * (target == classe))
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1, precision, recall, true_positive, false_positive, false_negative


def weighted_f1_score(
    output: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Calculate the weighted F1 score for a binary classification task.
    """
    _, predicted = torch.max(output, 1)

    assert predicted.shape == target.shape

    f1_list = torch.zeros(int(max(torch.unique(target))) + 1)
    weights = torch.zeros(int(max(torch.unique(target))) + 1)

    for classe in torch.unique(target):
        classe = int(classe)
        weight = len(target) / (2 * torch.sum(target == classe))
        weights[classe] = weight
        f1_list[classe] = f1_score(output, target, classe, epsilon)[0] * weight

    return torch.sum(f1_list) / torch.sum(weights)


def proba_f1_score(
    output: torch.Tensor,
    target: torch.Tensor,
    classe: torch.Tensor = torch.tensor(1),
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate the F1 score for a classification task.
    """

    assert classe <= output.shape[1]

    true_positive = (output * (target == classe).long().unsqueeze(1))[:, classe].sum()
    false_positive = (output * (target != classe).long().unsqueeze(1))[:, classe].sum()
    false_negative_tensor = output * (target == classe).long().unsqueeze(1)
    false_negative = torch.cat(
        (false_negative_tensor[:, :classe], false_negative_tensor[:, classe + 1 :]),
        dim=1,
    ).sum()
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1, precision, recall, true_positive, false_positive, false_negative


def weighted_proba_f1_score(
    output: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Calculate the weighted F1 score for a binary classification task.
    """
    _, predicted = torch.max(output, 1)

    assert predicted.shape == target.shape

    f1_list = torch.zeros(int(max(torch.unique(target))) + 1)
    weights = torch.zeros(int(max(torch.unique(target))) + 1)

    for classe in torch.unique(target):
        classe = int(classe)
        weight = len(target) / (2 * torch.sum(target == classe))
        weights[classe] = weight
        f1_list[classe] = proba_f1_score(output, target, classe, epsilon)[0] * weight

    return torch.sum(f1_list) / torch.sum(weights)


if __name__ == "__main__":

    dir_path = up(up(os.path.abspath(__file__)))
    a = torch.rand((10, 3))
    b = torch.randint(0, 3, (10,))
    print(a)
    print(b)
    print(weighted_proba_f1_score(a, b))
