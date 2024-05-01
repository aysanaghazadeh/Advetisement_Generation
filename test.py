import json
import pandas as pd

QA_original = json.load(open('../Data/PittAd/train/Action_Reason_statements.json'))
IDs = pd.read_csv('../Data/PittAd/train/simple_llava_description_all.csv')['ID'].values

# new_QA = {}
# for id in QA_original:
#     if id in IDs:
#         new_QA[id] = QA_original[id]
print(len(list(QA_original.keys())))
