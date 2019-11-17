import configparser
import logging
import os
import pathlib
import time
from multiprocessing.dummy import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from algo.distances import multi_run_wrapper_edit_distance
from algo.translation_scores import calculate_bleu_score, calculate_meteor_score
from util.logginghandler import LoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

configParser = configparser.RawConfigParser()
configFilePath = "config.txt"
configParser.read(configFilePath)

data_folder = configParser.get('data-preparing-config', 'DATA_FOLDER')
target_name = configParser.get('data-preparing-config', 'TARGET')
result_folder = configParser.get('result-config', 'RESULT_FOLDER')
# result_file = configParser.get('edit-distance-config', 'RESULT_FILE')
result_file = 'check.tsv'

print("Started reading the file")

src = pd.read_csv(os.path.join(data_folder, target_name, "src.tsv"), sep="\t")
target = pd.read_csv(os.path.join(data_folder, target_name, "target.tsv"), sep="\t")

full = pd.concat([src, target], axis=1)
full = full.head(100000)
print("Complete Number of rows {}".format(full.shape[0]))
train, test = train_test_split(full, test_size=0.2, random_state=777)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

print("Finished reading the file")

calculation_start = time.time()

bleu_scores = list()
meteor_scores = list()
distances = list()
retrieved_src_sentences = list()
retrieved_target_sentences = list()

test_src_sentences = test["source"].astype(str).tolist()
train_src_sentences = train["source"].astype(str).tolist()

pool = Pool(processes=13)

multi_processing_list = list()
for i in tqdm(range(len(test_src_sentences))):
    pair = [str(test_src_sentences[i]), train_src_sentences]
    multi_processing_list.append(pair)
    # sentence_distances = np.array(
    #     [edit_distance(str(test_src_sentences[i]), str(train_src_sentences[j])) for j in
    #      range(len(train_src_sentences))])
    #
    # sentence_distances = list()
    # for j in range(len(train_src_sentences)):
    #     distance = edit_distance(test_src_sentences[i], train_src_sentences[j])

# sentence_distances_list = pool.map(multi_run_wrapper_edit_distance, multi_processing_list)
sentence_distances_list = list(
    tqdm(pool.imap(multi_run_wrapper_edit_distance, multi_processing_list), total=len(multi_processing_list)))

for i in tqdm(range(len(sentence_distances_list))):
    sentence_distances = sentence_distances_list[i]
    # sentence_distances = edit_distances([str(test_src_sentences[i]), train_src_sentences])
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
