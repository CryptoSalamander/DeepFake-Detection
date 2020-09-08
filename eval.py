import json
import os
import csv
from glob import glob
from pathlib import Path

total = 0
correct = 0
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
FAKE_thres = 0.8
REAL_thres = 0.2
with open('output.json') as json_file:
    json_data = json.load(json_file)
    for csv_path in glob(os.path.join("./result", "*.csv")):
        dir = Path(csv_path).parent
        with open(csv_path, "r") as f:
            rdr = csv.reader(f)
            next(rdr)
            for line in rdr:
                total += 1 
                json_object = json_data[line[0]]
                if json_object['label'] == 'FAKE':
                    if float(line[1]) >= FAKE_thres:
                        correct += 1
                        true_positive += 1
                    else:
                        false_positive += 1
                elif json_object['label'] == 'REAL':
                    if float(line[1]) < REAL_thres:
                        correct += 1
                        true_negative += 1
                    else:
                        false_negative += 1
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
print('Accuracy \t',correct/total)
print('Precision\t', precision)
print('Recall\t\t', recall)
print('F1 Score\t', 2*(precision * recall) / (precision + recall))
print('Fall-out\t', false_positive / (true_negative + false_positive))
