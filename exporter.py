import hickle
import pickle

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