import xml


def get_sentences(path, source_language, target_language):
    source_sentences = list()
    target_sentences = list()
    doc = xml.dom.minidom.parse(path)
    items = doc.getElementsByTagName('tu')

    for elem in items:
        nested_items = elem.getElementsByTagName('tuv')
        source_sentence = None
        target_sentence = None
        for nested_item in nested_items:
            language = nested_item.attributes['lang'].value
            if language == source_language:
                source_sentence = nested_item.getElementsByTagName('seg')[0].firstChild.data
            if language == target_language:
                target_sentence = nested_item.getElementsByTagName('seg')[0].firstChild.data

        if target_sentence is not None and source_sentence is not None:
            source_sentences.append(source_sentence)
            target_sentences.append(target_sentence)

    return source_sentences, target_sentences
