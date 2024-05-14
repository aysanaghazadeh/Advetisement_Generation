import json
import pandas as pd

QA_original = json.load(open('../Data/PittAd/train/large_combined_hard_QA_Combined_Action_Reason_train.json'))
IDs = pd.read_csv('../Data/PittAd/train/simple_llava_description_all.csv')['ID'].values
#
count = 0
for image_url in IDs:
    if image_url not in QA_original:
        count += 1

print(count)
