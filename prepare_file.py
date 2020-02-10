import configparser
import os
import pathlib
from fnmatch import fnmatch

import pandas as pd

from parsing.xml_parser import get_sentences

configParser = configparser.RawConfigParser()
configFilePath = "config.txt"
configParser.read(configFilePath)

folder_path = configParser.get('data-preparing-config', 'FOLDER_PATH')
data_folder = configParser.get('data-preparing-config', 'DATA_FOLDER')
pattern = configParser.get('data-preparing-config', 'PATTERN')
source = configParser.get('data-preparing-config', 'SOURCE')
target = configParser.get('data-preparing-config', 'TARGET')

tmx_file_paths = [os.path.join(path, name) for path, subdirs, files in os.walk(folder_path) for name in files if
                  fnmatch(name, pattern)]

source_sentences = list()
target_sentences = list()

for tmx_file_path in tmx_file_paths:
    print(tmx_file_path)
    source_sentences_temp, target_sentences_temp = get_sentences(path=tmx_file_path, source_language=source,
                                                                 target_language=target)

    source_sentences = source_sentences + source_sentences_temp
    target_sentences = target_sentences + target_sentences_temp

assert (len(source_sentences) == len(target_sentences))

pathlib.Path(os.path.join(data_folder, target)).mkdir(parents=True, exist_ok=True)

src_data = pd.DataFrame(source_sentences, columns=['source'])
target_data = pd.DataFrame(target_sentences, columns=['target'])

src_data.to_csv(os.path.join(data_folder, target, "src_volume_1.tsv"), sep='\t', index=False, header=True)
target_data.to_csv(os.path.join(data_folder, target, "target_volume_1.tsv"), sep='\t', index=False, header=True)
