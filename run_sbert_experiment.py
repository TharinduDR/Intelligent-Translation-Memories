import configparser
import logging
import os
import pathlib
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from algo.embeddings.sbert import get_embeddings
from algo.translation_scores import calculate_bleu_score, calculate_meteor_score
from util.logginghandler import LoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

configParser = configparser.RawConfigParser()
configFilePath = "config.txt"
configParser.read(configFilePath)

data_folder = configParser.get('data-preparing-config', 'DATA_FOLDER')
target_name = configParser.get('data-preparing-config', 'TARGET')
result_folder = configParser.get('result-config', 'RESULT_FOLDER')
result_file = configParser.get('sbert-config', 'RESULT_FILE')

print("Started reading the file")

train_src = pd.read_csv(os.path.join(data_folder, target_name, "src_volume_1.tsv"), sep="\t")
train_target = pd.read_csv(os.path.join(data_folder, target_name, "target_volume_1.tsv"), sep="\t")

test_src = pd.read_csv(os.path.join(data_folder, target_name, "src_volume_3.tsv"), sep="\t")
test_target = pd.read_csv(os.path.join(data_folder, target_name, "target_volume_3.tsv"), sep="\t")

# full = pd.concat([src, target], axis=1)
# full = full.head(100000)
# train, test = train_test_split(full, test_size=0.2, random_state=777)

train = pd.concat([train_src, train_target], axis=1)
test = pd.concat([test_src, test_target], axis=1)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

print("Finished reading the file")

print("Getting Embeddings for {} sentences".format(train.shape[0] + test.shape[0]))

start = time.time()
train_embeddings = get_embeddings(train["source"].astype(str).tolist(), model)
test_embeddings = get_embeddings(test["source"].astype(str).tolist(), model)
end = time.time()
print("Finished getting embeddings from {} seconds".format(end - start))
print("Average time for a sentence is {}".format((end - start) / float(train.shape[0] + test.shape[0])))

time.sleep(1)
calculation_start = time.time()

bleu_scores = list()
meteor_scores = list()
distances = list()
retrieved_src_sentences = list()
retrieved_target_sentences = list()

for i in tqdm(range(len(test_embeddings))):
    all_distances = pairwise_distances([test_embeddings[i]], train_embeddings, metric='cosine', n_jobs=-1)
    sentence_distances = np.array([all_distances[0, j] for j in range(len(train_embeddings))])
    closestIdx = np.argmin(sentence_distances)
    closet_distance = np.amin(sentence_distances)
    retrieved_src_sentence = train["source"].astype(str).tolist()[closestIdx]
    retrieved_target_sentence = train["target"].astype(str).tolist()[closestIdx]

    bleu_score = calculate_bleu_score(test["target"].astype(str).tolist()[i], retrieved_target_sentence)
    meteor_score = calculate_meteor_score(test["target"].astype(str).tolist()[i], retrieved_target_sentence)
    retrieved_src_sentences.append(retrieved_src_sentence)
    retrieved_target_sentences.append(retrieved_target_sentence)
    distances.append(closet_distance)
    bleu_scores.append(bleu_score)
    meteor_scores.append(meteor_score)
    del all_distances

calculation_finish = time.time()

time.sleep(1)
print("Got similar sentences in {} seconds".format(calculation_finish - calculation_start))

test["retrieved_source_sentences"] = retrieved_src_sentences
test["distances"] = distances
test["retrieved_target_sentences"] = retrieved_target_sentences
test["bleu_scores"] = bleu_scores
test["meteor_scores"] = meteor_scores

pathlib.Path(os.path.join(result_folder, target_name)).mkdir(parents=True, exist_ok=True)

test.to_csv(os.path.join(result_folder, target_name, result_file), sep='\t', index=False)


