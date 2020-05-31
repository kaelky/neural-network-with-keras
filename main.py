import keras
import hickle
import numpy as np

from keras.models import load_model as load_model_from_keras
from sklearn.model_selection import train_test_split

from loader import load_raw_data, load_processed_data, construct_model
from loader import load_model_information
from exporter import save_training_data, save_testing_data, save_evaluation_result
from evaluator import get_report

data_version = "1"
#TODO(1) Uncomment line 14 until line 17 for saving the new version of training and testing data.
""" X, y = load_raw_data(f'/Users/kaelky/Workspace/Python/PDIB/discussion-07/data/raw/HMEQ-v{data_version}.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(0.40 * len(y)))
save_training_data(X_train, y_train)
save_testing_data(X_test, y_test) """


training_data = load_processed_data(f'/Users/kaelky/Workspace/Python/PDIB/discussion-07/data/processed/train_data.hkl')
testing_data = load_processed_data(f'/Users/kaelky/Workspace/Python/PDIB/discussion-07/data/processed/test_data.hkl')


X_train = training_data['X_train']
y_train = training_data['y_train']
X_test  = testing_data['X_test']
y_test  = testing_data['y_test']

loss_specs      = "binary_crossentropy"
optimizer_specs = "adam"
epochs_specs    = 2

model_version = "1"
#TODO(2) If you want to load the model from keras, uncomment line 37 until line 40
"""
model = load_model_from_keras(f"/Users/kaelky/Workspace/Python/PDIB/discussion-07/model/res/model_v{model_version}_data_v{data_version}_{loss_specs}_{optimizer_specs}_epochs={epochs_specs}.h5")
model.summary()
"""

model = construct_model(f'/Users/kaelky/Workspace/Python/PDIB/discussion-07/model/specs/v{model_version}.txt')
activation_function_from_input, activation_function_from_hidden = load_model_information(f'/Users/kaelky/Workspace/Python/PDIB/discussion-07/model/specs/v1.txt')

model.compile(loss=loss_specs, optimizer=optimizer_specs, metrics=['accuracy'])
model.fit(np.array(X_train), np.array(y_train), epochs=epochs_specs)
_, accuracy = model.evaluate(np.array(X_train), np.array(y_train))
print('Accuracy: %.2f' % (accuracy*100))


y_preds = model.predict(np.array(X_test))
normalized_preds = []
for pred in y_preds:
    normalized_preds.append(int(pred))

report = get_report(y_test, normalized_preds)

accuracy = report['accuracy']
precision = report['precision']
recall = report['recall']
f1 = report['f1']

print("Result of The Experiment")
print("========================")
print("Activation Function Detail Information")
print(f"Input Layer to Hidden Layer  : {activation_function_from_input}")
print(f"Hidden Layer to Output Layer: {activation_function_from_hidden}")
print("========================")
print(f"Accuracy     : {accuracy}")
print(f"Precision    : {precision}")
print(f"Recall       : {recall}")
print(f"F1-Measure   : {f1}")

report_detail = f"model_v{model_version}_data_v{data_version}_{loss_specs}_{optimizer_specs}_epochs_{epochs_specs}"
save_evaluation_result(report, report_detail)

#TODO(3) If you want to save the model to keras, uncomment line 69
model.save(f"/Users/kaelky/Workspace/Python/PDIB/discussion-07/model/res/model_v{model_version}_data_v{data_version}_{loss_specs}_{optimizer_specs}_epochs={epochs_specs}.h5")