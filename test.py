import json
import pandas as pd

llama_file = json.load(open('/Users/aysanaghazadeh/experiments/results/llama3_FT_generated_description_new_test_setGPT4v_persuasiveness_alignment.json_SDXL_20240616_232723GPT4v_persuasiveness_alignment.json'))
# ar_file = json.load(open('/Users/aysanaghazadeh/experiments/results/AR_PixArt_20240505_231631.json'))

values = []
above_5_count = 0
below_neg_5_count = 0
print(len(llama_file))
for filename in list(llama_file.keys()):
    if llama_file[filename] > 5:
        above_5_count += 1
    elif llama_file[filename] < -5:
        below_neg_5_count += 1
    else:
        values.append(llama_file[filename])
print(sum(values)/len(values))
print(above_5_count)
print(below_neg_5_count)
# for key in ar_file:
#     if key in llama_file:
#         values.append(llama_file[key])
# print(len(ar_file))
# print(sum(values)/len(values))
# print(sum(ar_file.values())/len(ar_file))
print(sum(list(llama_file.values()))/len(list(llama_file.values())))
# max_key = max(llama_file, key=llama_file.get)
# print(max_key)
# max_key = max(ar_file, key=ar_file.get)
# print(max_key)
#
# print(llama_file[max_key])
# print(ar_file[max_key])
# ranking = json.load(open('/Users/aysanaghazadeh/experiments/results/LLAMA3_generated_prompt_PixArt_20240508_084149_image_text_ranking.json'))

# print(sum(ranking.values())/len(ranking))


images = [
    "160840.jpg",
    "135670.jpg",
    "132590.jpg",
    "54690.jpg",
    "119260.jpg",
    "24230.jpg",
    "66310.jpg",
    "130250.jpg",
    "84000.jpg",
    "119350.jpg",
    "89100.jpg",
    "1130.jpg",
    "100430.jpg",
    "103430.jpg",
    "22540.jpg",
    "75950.jpg",
    "29960.jpg",
    "58210.jpg",
    "101220.jpg",
    "13530.jpg",
    "121200.jpg",
    "55630.jpg",
    "130060.jpg",
    "87020.jpg",
    "99570.jpg",
    "149830.jpg",
    "111040.jpg",
    "136200.jpg",
    "136560.jpg",
    "135640.jpg",
    "94080.jpg",
    "159800.jpg",
    "57870.jpg",
    "68540.jpg",
    "67470.jpg",
    "25260.jpg",
    "81460.jpg"
]

# for image in images:
#     image_url = '0/' + image
#     if llama_file[image_url] > ar_file[image_url] and abs(llama_file[image_url] - ar_file[image_url]) >= 0.005:
#         print(0, llama_file[image_url] - ar_file[image_url])
#     if llama_file[image_url] == ar_file[image_url] or abs(llama_file[image_url] - ar_file[image_url]) < 0.005:
#         print('-', llama_file[image_url] - ar_file[image_url])
#     if llama_file[image_url] < ar_file[image_url] and abs(llama_file[image_url] - ar_file[image_url]) >= 0.005:
#         print(1, ar_file[image_url] - llama_file[image_url])


persuasiveness = json.load(open('/Users/aysanaghazadeh/experiments/results/persuasiveness.json'))
print(len(persuasiveness))
print(sum(list(persuasiveness.values())[0:5000])/len(list(persuasiveness.values())[0:5000]))
import os
root_directory = '../Data/PittAd/test_set'
count = 0


