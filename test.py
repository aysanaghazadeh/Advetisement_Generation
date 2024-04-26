import pandas as pd
import os
import shutil
import json

test_images_list = '/Users/aysanaghazadeh/test_image.csv'
test_images_list = pd.read_csv(test_images_list)['ID'].values
# os.mkdir('../Data/PittAd/test_set')
QA = '../Data/PittAd/train/large_combined_hard_QA_Combined_Action_Reason_train.json'
QA = json.load(open(QA))
count = 0
for image in test_images_list:
    if image in QA:
        count += 1
        print(image)

print(count)

