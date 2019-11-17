import configparser
import os
import pathlib

import numpy as np
import pandas as pd

from algo.translation_scores import calculate_bleu_score
from parsing.tmx import tmxfile

configParser = configparser.RawConfigParser()
configFilePath = "config.txt"
configParser.read(configFilePath)

source = configParser.get('data-preparing-config', 'SOURCE')
target = configParser.get('data-preparing-config', 'TARGET')
data_folder = configParser.get('data-preparing-config', 'DATA_FOLDER')
target_name = configParser.get('data-preparing-config', 'TARGET')
result_folder = configParser.get('result-config', 'RESULT_FOLDER')
result_file = configParser.get('okapi-config', 'RESULT_FILE')

test_src = pd.read_csv(os.path.join(data_folder, target_name, "src_volume_3.tsv"), sep="\t")
test_target = pd.read_csv(os.path.join(data_folder, target_name, "target_volume_3.tsv"), sep="\t")

test = pd.concat([test_src, test_target], axis=1)
test = test.reset_index(drop=True)

print(test.head())

test_source_sentences = test["source"].astype(str).tolist()
test_target_sentences = test["target"].astype(str).tolist()

source_sentences = list()
target_sentences = list()
retrived_target_sentences = list()
bleu_scores = list()

tmx_file_path = "result/ES-ES/unapproved.tmx"

with open(tmx_file_path, 'rb') as fin:
    tmx_file = tmxfile(fin, source, target)
    i = 0
    for node in tmx_file.unit_iter():
        i = i + 1
        source_sentence = node.getsource().strip()
        retrieved_target_sentence = node.gettarget().strip()

        index = test_source_sentences.index(source_sentence) if source_sentence in test_source_sentences else -1

        if index > -1:
            target_sentence = test_target_sentences[index]
            bleu_score = calculate_bleu_score(target_sentence, retrieved_target_sentence)
            source_sentences.append(source_sentence)
            target_sentences.append(target_sentence)
            retrived_target_sentences.append(retrieved_target_sentence)
            bleu_scores.append(bleu_score)

        if i % 10 == 0:
            print(i)

    fin.close()

print(target_sentences)

# print(len(source_sentences))
# print(len(target_sentences))
# print(len(retrived_target_sentences))
# print(len(bleu_scores))
#
# print("started saving")
#
# with open('some.csv', 'w') as f:
#     writer = csv.writer(f, delimiter='\t')
#     writer.writerows(zip(source_sentences, target_sentences))

okapi_results = pd.DataFrame(
    np.column_stack([source_sentences, target_sentences, retrived_target_sentences, bleu_scores]),
    columns=['source', 'target', 'retrieved_target_sentences', 'bleu_scores'])

pathlib.Path(os.path.join(result_folder, target_name)).mkdir(parents=True, exist_ok=True)
okapi_results.to_csv(os.path.join(result_folder, target_name, result_file), sep='\t', index=False)
#
# okapi_results = pd.DataFrame(source_sentences, columns=['source'])
#
# pathlib.Path(os.path.join(result_folder, target_name)).mkdir(parents=True, exist_ok=True)
# okapi_results.to_csv(os.path.join(result_folder, target_name, result_file), sep='\t', index=False)
#
# okapi_results_temp = pd.read_csv(os.path.join(result_folder, target_name, result_file), sep="\t")
# #
# print("adding more columns")
# #
# okapi_results_temp["target"] = pd.Series(target_sentences)
# # okapi_results['retrieved_target_sentences'] = retrived_target_sentences
# # okapi_results['bleu_scores'] = bleu_scores
#
# # okapi_results_temp.to_csv(os.path.join(result_folder, target_name, result_file), sep='\t', index=False)

print("done")
