from typing import List


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """Compute recall at k.

    Args:
        y_true: list of true labels
        y_pred: list of predicted labels
        k: number of top labels to consider

    Returns:
        Recall at k
    """
    score_label = sorted(zip(scores, labels), key=lambda t: t[0], reverse=True)
    label_sort = [x[1] for x in score_label]
    tp = sum(label_sort[:k])
    fn = sum(label_sort[k:])
    return tp/(tp+fn)


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """Compute precision at k.

    Args:
        y_true: list of true labels
        y_pred: list of predicted labels
        k: number of top labels to consider

    Returns:
        Precision at k
    """
    score_label = sorted(zip(scores, labels), key=lambda t: t[0], reverse=True)
    label_sort = [x[1] for x in score_label]
    tp = sum(label_sort[:k])
    fp = k-tp
    return tp/(fp+tp)


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """Compute specificity at k.

    Args:
        y_true: list of true labels
        y_pred: list of predicted labels
        k: number of top labels to consider

    Returns:
        Specificity at k
    """
    score_label = sorted(zip(scores, labels), key=lambda t: t[0], reverse=True)
    label_sort = [x[1] for x in score_label]
    tp = sum(label_sort[:k])
    fp = k-tp
    fn = sum(label_sort[k:])
    tn = len(label_sort) - k - fn
    if(fp+tn) == 0:
        return 0
    return tn/(fp+tn)


def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """Compute f1 score at k.

    Args:
        y_true: list of true labels
        y_pred: list of predicted labels
        k: number of top labels to consider

    Returns:
        F1 score at k
    """
    precision = precision_at_k(labels, scores, k)
    recall = recall_at_k(labels, scores, k)
    if(precision+recall) == 0:
        return 0
    return 2*precision*recall/(precision + recall)
