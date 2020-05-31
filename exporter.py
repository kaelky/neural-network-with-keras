import hickle
import pickle
import csv

def save_training_data(X_train, y_train):
    hickle.dump({
        'X_train': X_train,
        'y_train': y_train,
    }, f'data/processed/train_data.hkl', mode='w')

def save_testing_data(X_test, y_test):
    hickle.dump({
        'X_test' : X_test,
        'y_test' : y_test,
    }, f'data/processed/test_data.hkl', mode='w')

def save_evaluation_result(report, report_detail):
    rows = [
        ["Accuracy", report['accuracy']],
        ["Precision", report['precision']],
        ["Recall", report['recall']],
        ["f1", report['f1']],
    ]

    with open(f'/Users/kaelky/Workspace/Python/PDIB/discussion-07/generated-result/{report_detail}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)