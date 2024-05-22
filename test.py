import json
import pandas as pd

persuasiveness_file = '/Users/aysanaghazadeh/experiments/results/LLAMA3_generated_prompt_PixArt_20240508_084149_image_text_alignment.json'
persuasiveness = json.load(open(persuasiveness_file))
print(len(persuasiveness))
print(sum(persuasiveness.values())/len(persuasiveness.values()))

