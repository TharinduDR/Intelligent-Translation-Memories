from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score

smoother = SmoothingFunction()


def calculate_bleu_score(reference, hypothesis):
    return sentence_bleu([reference.split()], hypothesis.split(), weights=(0.5, 0.5),
                         smoothing_function=smoother.method1)


def calculate_meteor_score(reference, hypothesis):
    return single_meteor_score(reference, hypothesis)
