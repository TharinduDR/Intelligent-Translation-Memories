import configparser
import os
import pathlib

import numpy as np
import pandas as pd

configParser = configparser.RawConfigParser()
configFilePath = "config.txt"
configParser.read(configFilePath)

target_name = configParser.get('data-preparing-config', 'TARGET')
result_folder = configParser.get('result-config', 'RESULT_FOLDER')

infersent_result = pd.read_csv("result/ES-ES/infersent.tsv", sep="\t")

infersent_source_sentences = infersent_result["source"].astype(str).tolist()
infersent_retrieved_sentences = infersent_result["retrieved_target_sentences"].astype(str).tolist()
infersent_bleu_scores = infersent_result["bleu_scores"].tolist()

sbert_result = pd.read_csv("result/ES-ES/sbert.tsv", sep="\t")

sbert_source_sentences = sbert_result["source"].astype(str).tolist()
sbert_retrieved_sentences = sbert_result["retrieved_target_sentences"].astype(str).tolist()
sbert_bleu_scores = sbert_result["bleu_scores"].tolist()

universal_encoder_result = pd.read_csv("result/ES-ES/universal_encoder.tsv", sep="\t")

universal_encoder_source_sentences = universal_encoder_result["source"].astype(str).tolist()
universal_encoder_retrieved_sentences = universal_encoder_result["retrieved_target_sentences"].astype(str).tolist()
universal_encoder_bleu_scores = universal_encoder_result["bleu_scores"].tolist()

okapi_results = pd.read_csv("result/ES-ES/okapi.tsv", sep="\t")
okapi_source_sentences = okapi_results["source"].astype(str).tolist()
okapi_target_sentences = okapi_results["target"].astype(str).tolist()
okapi_retrieved_sentences = okapi_results["retrieved_target_sentences"].astype(str).tolist()
okapi_bleu_scores = okapi_results["bleu_scores"].tolist()

infersent_sentences = list()
infersent_scores = list()

sbert_sentences = list()
sbert_scores = list()

universal_encoder_sentences = list()
universal_encoder_scores = list()

okapi_sentences = list()
okapi_scores = list()

source_sentences = list()
target_sentences = list()

i = 0
for source_sentence in okapi_source_sentences:
    index = infersent_source_sentences.index(source_sentence) if source_sentence in infersent_source_sentences else -1
    if index > -1:
        target_sentence = okapi_target_sentences[i]
        infersent_sentence = infersent_retrieved_sentences[index]
        infersent_score = infersent_bleu_scores[index]

        sbert_sentence = sbert_retrieved_sentences[index]
        sbert_score = sbert_bleu_scores[index]

        universal_encoder_sentence = universal_encoder_retrieved_sentences[index]
        universal_encoder_score = universal_encoder_bleu_scores[index]

        okapi_sentence = okapi_retrieved_sentences[i]
        okapi_score = okapi_bleu_scores[i]

        source_sentences.append(source_sentence)
        target_sentences.append(target_sentence)
        infersent_sentences.append(infersent_sentence)
        infersent_scores.append(infersent_score)
        sbert_sentences.append(sbert_sentence)
        sbert_scores.append(sbert_score)
        universal_encoder_sentences.append(universal_encoder_sentence)
        universal_encoder_scores.append(universal_encoder_score)
        okapi_sentences.append(okapi_sentence)
        okapi_scores.append(okapi_score)

    i = i + 1
    if i % 10 == 0:
        print(i)

full_results = pd.DataFrame(
    np.column_stack(
        [source_sentences, target_sentences, infersent_sentences, sbert_sentences, universal_encoder_sentences,
         okapi_sentences, infersent_scores,
         sbert_scores, universal_encoder_scores, okapi_scores]),
    columns=['source', 'target', 'infersent_sentences', 'sbert_sentences', 'universal_encoder_sentences',
             'okapi_sentences', 'infersent_scores',
             'sbert_scores', 'universal_encoder_scores', 'okapi_scores'])

pathlib.Path(os.path.join(result_folder, target_name)).mkdir(parents=True, exist_ok=True)
full_results.to_csv(os.path.join(result_folder, target_name, "full_results.tsv"), sep='\t', index=False)

print("done")
