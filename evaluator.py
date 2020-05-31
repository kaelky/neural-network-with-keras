def get_report(truths, predictions):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for truth, prediction in zip(truths, predictions):
        if truth == 0 and prediction == 0:
            tp += 1
        elif truth == 0 and prediction == 1:
            fn += 1
        elif truth == 1 and prediction == 0:
            fp += 1
        elif truth == 1 and prediction == 1:
            tn += 1

    accuracy    = round((tp + tn) / (tp + fp + tn + fn), 4)
    precision   = round(tp / (tp + fp), 4)
    recall      = round(tp / (tp + fn), 4)
    f1_measure  = round(2 * ((precision * recall)/(precision + recall)), 4)

    report = {
        'accuracy'  : accuracy,
        'precision' : precision,
        'recall'    : recall,
        'f1'        : f1_measure,
    }

    return report