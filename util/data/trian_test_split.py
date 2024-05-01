import os
import pandas as pd
import random
import csv
import json
from collections import Counter, defaultdict

topic_map = {1: ["Restaurants", "cafe", "fast food"],
             2: ["Chocolate", "cookies", "candy", "ice cream"],
             3: ["Chips", "snacks", "nuts", "fruit", "gum", "cereal", "yogurt", "soups"],
             4: ["Seasoning", "condiments", "ketchup"],
             5: ["Pet food"],
             6: ["Alcohol"],
             7: ["Coffee", "tea"],
             8: ["Soda", "juice", "milk", "energy drinks", "water"],
             9: ["Cars", "automobiles", "car sales", "auto parts", "car insurance", "car repair", "gas", "motor oil"],
             10: ["Electronics", "computers", "laptops", "tablets", "cellphones", "TVs"],
             11: ["Phone", "TV and internet service providers"],
             12: ["Financial services", "banks", "credit cards", "investment firms"],
             13: ["Education", "universities", "colleges", "kindergarten", "online degrees"],
             14: ["Security and safety services", "anti-theft", "safety courses"],
             15: ["Software", "internet radio", "streaming", "job search website", "grammar correction",
                  "travel planning"],
             16: ["dating", "tax", "legal", "loan", "religious", "printing", "catering"],
             17: ["Beauty products and cosmetics"],
             18: ["Healthcare and medications", "hospitals", "health insurance", "allergy", "cold remedy", "home tests",
                  "vitamins"],
             19: ["Clothing and accessories", "jeans", "shoes", "eye glasses", "handbags", "watches", "jewelry"],
             20: ["Baby products", "baby food", "sippy cups", "diapers"],
             21: ["Games and toys", "including video and mobile games"],
             22: ["Cleaning products", "detergents", "fabric softeners", "soap", "tissues", "paper towels"],
             23: ["Home improvements and repairs", "furniture", "decoration", "lawn care", "plumbing"],
             24: ["Home appliances", "coffee makers", "dishwashers", "cookware", "vacuum cleaners", "heaters",
                  "music players"],
             25: ["Vacation and travel", "airlines", "cruises", "theme parks", "hotels", "travel agents"],
             26: ["Media and arts", "TV shows", "movies", "musicals", "books", "audio books"],
             27: ["Sports equipment and activities"],
             28: ["Shopping", "department stores", "drug stores", "groceries"],
             29: ["Gambling", "lotteries", "casinos"],
             30: ["Environment", "nature", "pollution", "wildlife"],
             31: ["Animal rights", "animal abuse"],
             32: ["Human rights"],
             33: ["Safety", "safe driving", "fire safety"],
             34: ["Smoking", "alcohol abuse"],
             35: ["Domestic violence"],
             36: ["Self esteem", "bullying", "cyber bullying"],
             37: ["Political candidates", "support or opposition"],
             38: ["Charities"],
             39: ["Unclear"]}


def get_train_data(args):
    train_file = os.path.join(args.data_path, 'train/train_image.csv')
    if os.path.exists(train_file):
        return pd.read_csv(train_file).values
    QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    image_urls = list(QA.keys())
    train_size = int(0.33 * len(image_urls))
    train_image_urls = random.sample(image_urls, train_size)
    print('saving train data')
    with open(train_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID'])

        # Write the data
        for i in train_image_urls:
            writer.writerow([i])
    return pd.read_csv(train_file).values


def get_test_data(args):
    topics_data_file = os.path.join(args.data_path, 'train/Topics_train.json')
    test_file = os.path.join(args.data_path, 'train/test_image.csv')
    if os.path.exists(test_file):
        return pd.read_csv(test_file)
    topics_data = json.load(open(topics_data_file))
    all_topics = [topic for topics in topics_data.values() for topic in set(topics)]
    topic_counter = Counter(all_topics)
    most_common_topics = [topic for topic, count in topic_counter.most_common(10)]
    selected_files = defaultdict(list)
    train_files = get_train_data(args)
    QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    for file, topics in topics_data.items():
        if file in train_files or file not in QA:
            continue
        for topic in set(topics):
            if topic in most_common_topics:
                if int(topic) in topic_map:
                    if len(selected_files[topic]) < 300:
                        selected_files[topic].append(file)
    print('saving test files...')
    with open(test_file, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['ID', 'topic'])

        for topic, files in selected_files.items():
            for filename in files:
                writer.writerow([filename, '-'.join(topic_map[int(topic)])])
    return pd.read_csv(test_file)

