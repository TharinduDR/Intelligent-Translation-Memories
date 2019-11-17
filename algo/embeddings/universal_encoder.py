import warnings

import tensorflow_hub as hub

warnings.simplefilter(action='ignore', category=FutureWarning)

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

# Import the Universal Sentence Encoder's TF Hub module

embed = hub.Module(module_url)


def get_embeddings(sentences, session):
    embeddings = session.run(embed(sentences))
    return embeddings
