import sys

from nltk import edit_distance


def edit_distances(test_instance, training_instances):
    distances = []
    exact_match = False
    for j in range(len(training_instances)):
        if not exact_match:
            distance = edit_distance(test_instance, training_instances[j])
        else:
            distance = sys.maxsize
        distances.append(distance)
        if distance == 0:
            exact_match = True
    return distances


# def edit_distances_wrapper(sentence_pair):
#     return edit_distances(sentence_pair[0], sentence_pair[1])

def multi_run_wrapper_edit_distance(args):
    return edit_distances(*args)
