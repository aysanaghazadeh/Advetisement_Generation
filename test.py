import pandas as pd
import os
import shutil

test_images_list = '/Users/aysanaghazadeh/test_image.csv'
test_images_list = pd.read_csv(test_images_list)['ID'].values
# os.mkdir('../Data/PittAd/test_set')
images = '../Data/AdData/train_images'
for filename in test_images_list:
    if not os.path.exists('../Data/PittAd/test_set/'+filename.split('/')[0]):
        os.mkdir('../Data/PittAd/test_set/'+filename.split('/')[0])
    shutil.copyfile(os.path.join(images, filename), '../Data/PittAd/test_set/'+filename)
