import json
import pandas as pd

persuasiveness_file = '/Users/aysanaghazadeh/experiments/results/LLAMA3_generated_prompt_PixArt_20240508_084149.json'
persuasiveness = json.load(open(persuasiveness_file))

print(sum(persuasiveness.values())/len(persuasiveness.values()))

