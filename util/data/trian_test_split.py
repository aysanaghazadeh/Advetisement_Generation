import os
import pandas as pd
import random
import csv
import json
from collections import Counter, defaultdict
from util.data.mapping import TOPIC_MAP as topic_map


def get_train_data(args):
    train_file = os.path.join(args.data_path, 'train/train_image.csv')
    if os.path.exists(train_file):
        return pd.read_csv(train_file).values
    QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    image_urls = list(QA.keys())
    print(len(image_urls))
    train_size = int(args.train_ratio * len(image_urls))
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

