import json
import pandas as pd

llama_file = json.load(open('/Users/aysanaghazadeh/experiments/results/LLAMA3_generated_prompt_PixArt_20240508_084149_image_text_alignment.json'))
ar_file = json.load(open('/Users/aysanaghazadeh/experiments/results/AR_PixArt_20240505_231631_image_text_alignment.json'))

values = []
for key in ar_file:
    if key in llama_file:
        values.append(llama_file[key])
print(len(ar_file))
print(sum(values)/len(values))
print(sum(ar_file.values())/len(ar_file))
print(sum(list(llama_file.values())[664:])/len(list(llama_file.values())[664:]))


ranking = json.load(open('/Users/aysanaghazadeh/experiments/results/LLAMA3_generated_prompt_PixArt_20240508_084149_image_text_ranking.json'))

print(sum(ranking.values())/len(ranking))