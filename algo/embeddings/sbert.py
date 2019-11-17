def get_embeddings(sentences, model):
    embeddings = model.encode(sentences, batch_size=128)
    return embeddings
