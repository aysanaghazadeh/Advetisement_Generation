import os.path
import pandas as pd
import json
from collections import Counter
from Evaluation.metrics import Metrics, PersuasivenessMetric
from configs.inference_config import get_args

TOPIC_FILE = '../Data/PittAd/train/Topics_train.json'
RESULT_FILE = '../experiments/results/AR_PixArt_20240505_231631.csv'
topic_map = {'1': ["Restaurants", "cafe", "fast food"],
             '2': ["Chocolate", "cookies", "candy", "ice cream"],
             '3': ["Chips", "snacks", "nuts", "fruit", "gum", "cereal", "yogurt", "soups"],
             '4': ["Seasoning", "condiments", "ketchup"],
             '5': ["Pet food"],
             '6': ["Alcohol"],
             '7': ["Coffee", "tea"],
             '8': ["Soda", "juice", "milk", "energy drinks", "water"],
             '9': ["Cars", "automobiles", "car sales", "auto parts", "car insurance", "car repair", "gas", "motor oil"],
             '10': ["Electronics", "computers", "laptops", "tablets", "cellphones", "TVs"],
             '11': ["Phone", "TV and internet service providers"],
             '12': ["Financial services", "banks", "credit cards", "investment firms"],
             '13': ["Education", "universities", "colleges", "kindergarten", "online degrees"],
             '14': ["Security and safety services", "anti-theft", "safety courses"],
             '15': ["Software", "internet radio", "streaming", "job search website", "grammar correction",
                  "travel planning"],
             '16': ["dating", "tax", "legal", "loan", "religious", "printing", "catering"],
             '17': ["Beauty products and cosmetics"],
             '18': ["Healthcare and medications", "hospitals", "health insurance", "allergy", "cold remedy", "home tests",
                  "vitamins"],
             '19': ["Clothing and accessories", "jeans", "shoes", "eye glasses", "handbags", "watches", "jewelry"],
             '20': ["Baby products", "baby food", "sippy cups", "diapers"],
             '21': ["Games and toys", "including video and mobile games"],
             '22': ["Cleaning products", "detergents", "fabric softeners", "soap", "tissues", "paper towels"],
             '23': ["Home improvements and repairs", "furniture", "decoration", "lawn care", "plumbing"],
             '24': ["Home appliances", "coffee makers", "dishwashers", "cookware", "vacuum cleaners", "heaters",
                  "music players"],
             '25': ["Vacation and travel", "airlines", "cruises", "theme parks", "hotels", "travel agents"],
             '26': ["Media and arts", "TV shows", "movies", "musicals", "books", "audio books"],
             '27': ["Sports equipment and activities"],
             '28': ["Shopping", "department stores", "drug stores", "groceries"],
             '29': ["Gambling", "lotteries", "casinos"],
             '30': ["Environment", "nature", "pollution", "wildlife"],
             '31': ["Animal rights", "animal abuse"],
             '32': ["Human rights"],
             '33': ["Safety", "safe driving", "fire safety"],
             '34': ["Smoking", "alcohol abuse"],
             '35': ["Domestic violence"],
             '36': ["Self esteem", "bullying", "cyber bullying"],
             '37': ["Political candidates", "support or opposition"],
             '38': ["Charities"],
             '39': ["Unclear"]}


def get_topic_based_results():
    topics_data = json.load(open(TOPIC_FILE))
    all_topics = [topic for topics in topics_data.values() for topic in set(topics)]
    topic_counter = Counter(all_topics)
    most_common_topics = [topic for topic, count in topic_counter.most_common(10)]
    results = pd.read_csv(RESULT_FILE).values
    FIDs = {}
    CLIP_scores = {}
    image_topics = json.load(open(TOPIC_FILE))
    for row in results:
        image_url = row[0]
        FID = row[7]
        CLIP_score = row[5]
        topics = image_topics[image_url]
        for topic in topics:
            if topic in topic_map and topic in most_common_topics:
                topic_string = '-'.join(topic_map[topic])
                if topic_string in FIDs:
                    FIDs[topic_string].append(FID)
                    CLIP_scores[topic_string].append(CLIP_score)
                else:
                    FIDs[topic_string] = [FID]
                    CLIP_scores[topic_string] = [CLIP_score]

    for topic in FIDs:
        print(f'Average FID for {topic} is: {sum(FIDs[topic])/len(FIDs[topic])}')
        print(f'Average CLIP-score for {topic} is: {sum(CLIP_scores[topic])/len(CLIP_scores[topic])}')
        print('*'*80)

def evaluate_results(metrics, args):
    results = pd.read_csv(RESULT_FILE).values
    FIDs = []
    CLIP_scores = []
    QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    for row in results:
        image_url = row[0]
        FID = row[-1]
        CLIP_score = row[5]
        action_reason = QA[image_url][0]
        original_image_url = os.path.join(args.data_path, args.test_set_images, image_url)
        original_image_text_score = metrics.get_text_image_CLIP_score(original_image_url, action_reason, args)['text']
        if original_image_text_score > 0.23:
            FIDs.append(FID)
            CLIP_scores.append(CLIP_score)
        else:
            print(f'Text image score for image {image_url} is {original_image_text_score}')
            print('-'*100)
    print(f'number of examples: {len(FIDs)}')
    print(f'Average FID is: {sum(FIDs) / len(FIDs)}')
    print(f'Average CLIP-score is: {sum(CLIP_scores) / len(CLIP_scores)}')
    print('*' * 80)


def evaluate_persuasiveness():
    persuasiveness = PersuasivenessMetric()
    saving_path = RESULT_FILE.replace('.csv', '.json')
    print(saving_path)
    results = pd.read_csv(RESULT_FILE).values
    persuasiveness_scores = {}
    for row in results:
        image_url = row[0]
        generated_image_path = row[3]
        persuasiveness_score = persuasiveness.get_persuasiveness_score(generated_image_path)
        print(f'persuasiveness score of the image {image_url} is {persuasiveness_score} out of 10')
        print('*' * 80)
        persuasiveness_scores[image_url] = persuasiveness_score
        with open(saving_path, "w") as outfile:
            json.dump(persuasiveness_scores, outfile)
    print(f'average persuasiveness is {sum(persuasiveness_scores)/len(persuasiveness_scores)}')
    with open(saving_path, "w") as outfile:
        json.dump(persuasiveness_scores, outfile)

if __name__ == '__main__':
    args = get_args()
    metrics = Metrics(args)
    # evaluate_results(metrics, args)
    # get_topic_based_results()
    evaluate_persuasiveness()
