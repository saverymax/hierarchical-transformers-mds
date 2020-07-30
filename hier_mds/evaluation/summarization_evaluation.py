"""
Script to run summarization evaluation, independently of chiqa-fsl-ks repository. Uses code from first answer summ project.
"""

import argparse
import csv
import pandas as pd
import sys
import json
import collections
from collections import Counter
import statistics
import math
import nltk
import numpy as np
import scipy
import warnings

from tabulate import tabulate

def get_args():
    """
    Argument defnitions
    """
    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("--pred_dir",
                        default="",
                        dest="pred_dir",
                        help="Directory to read predicted (generated) text")
    parser.add_argument("--per_sample",
                        action="store_true",
                        dest="per_sample",
                        help="Compute rouge2 and BLEU per sample")
    parser.add_argument("--per_sample",
                        action="store_true",
                        dest="per_sample",
                        help="Compute rouge2 and BLEU per sample")
    parser.add_argument("--error_est",
                        action="store_true",
                        dest="error_estimates",
                        help="Show the mean and std. dev. for a set of repeated trials")
    parser.add_argument("--extractiveness",
                        action="store_true",
                        dest="extractiveness",
                        help="Compute the extractiveness of the summary to the source article")
    parser.add_argument("--bootstrap",
                        action="store_true",
                        dest="bootstrap",
                        help="Bootstrap evaluation for 1000 iterations. Report confidence intervals of rouge scores")
    parser.add_argument("--wilcoxon",
                        dest="calculate_wilcoxon",
                        action="store_true",
                        help="Compute wilcoxn statistics between rouge scores of models")
    return parser

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _split_into_words(sentences):
    # Set up for if there are multiple sentences for one reference, but we just have one: 
    # in format [["This", "is", "the", "sentence"]]
    assert isinstance(sentences[0], list)
    assert isinstance(sentences[0][0], str)

    return [word for sentence in sentences for word in sentence]


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)


def _len_lcs(x, y):
    """
    Returns the length of the Longest Common Subsequence between sequences x
    and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns
      integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: collection of words
      y: collection of words

    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _recon_lcs(x, y):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = _lcs(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
    return recon_tuple


def bleu_3_7(hypotheses, references):
    """Calculates average rouge scores for a list of hypotheses and
    references"""
    rouge_dict = {}
    for i in [3, 4, 5, 6, 7]:
        rouge_per_summ = [rouge_n([hyp], [ref], i, rouge=False) for hyp, ref in zip(hypotheses, references)]
        rouge_dict['rouge_{}'.format(i)] = np.mean(rouge_per_summ)
    print(rouge_dict)
    rouges = [rouge_dict[r] for r in rouge_dict]
    geo_mean = np.prod(rouges) ** (1 / len(rouges))

    return geo_mean

 
def rouge_n(evaluated_sentences, reference_sentences, n=2, rouge=True):
    """
    Computes ROUGE-N of two text collections of sentences.
    Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf

    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentences: The sentences from the referene set
      n: Size of ngram.  Defaults to 2.

    Returns:
      A tuple (f1, precision, recall) for ROUGE-N

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    # return overlapping_count / reference_count
    if rouge:
        return f1_score, precision, recall
    else:
        return precision


def _f_p_r_lcs(llcs, m, n):
    """
    Computes the LCS-based F-measure score
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Args:
      llcs: Length of LCS
      m: number of words in reference summary
      n: number of words in candidate summary

    Returns:
      Float. LCS-based F-measure score
    """
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta ** 2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta ** 2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return f_lcs, p_lcs, r_lcs


def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
    """
    Computes ROUGE-L (sentence level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Calculated according to:
    R_lcs = LCS(X,Y)/m
    P_lcs = LCS(X,Y)/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

    where:
    X = reference summary
    Y = Candidate summary
    m = length of reference summary
    n = length of candidate summary

    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentences: The sentences from the referene set

    Returns:
      A float: F_lcs

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")
    reference_words = _split_into_words(reference_sentences)
    evaluated_words = _split_into_words(evaluated_sentences)
    m = len(reference_words)
    n = len(evaluated_words)
    lcs = _len_lcs(evaluated_words, reference_words)
    return _f_p_r_lcs(lcs, m, n)


def _union_lcs(evaluated_sentences, reference_sentence):
    """
    Returns LCS_u(r_i, C) which is the LCS score of the union longest common
    subsequence between reference sentence ri and candidate summary C. For example
    if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8 and
    c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1 is
    "w1 w2" and the longest common subsequence of r_i and c2 is "w1 w3 w5". The
    union longest common subsequence of r_i, c1, and c2 is "w1 w2 w3 w5" and
    LCS_u(r_i, C) = 4/5.

    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentence: One of the sentences in the reference summaries

    Returns:
      float: LCS_u(r_i, C)

    ValueError:
      Raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    lcs_union = set()
    reference_words = _split_into_words([reference_sentence])
    combined_lcs_length = 0
    for eval_s in evaluated_sentences:
        evaluated_words = _split_into_words([eval_s])
        lcs = set(_recon_lcs(reference_words, evaluated_words))
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union.union(lcs)

    union_lcs_count = len(lcs_union)
    union_lcs_value = union_lcs_count / combined_lcs_length
    return union_lcs_value


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
    """
    Computes ROUGE-L (summary level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Calculated according to:
    R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
    P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

    where:
    SUM(i,u) = SUM from i through u
    u = number of sentences in reference summary
    C = Candidate summary made up of v sentences
    m = number of words in reference summary
    n = number of words in candidate summary

    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentences: One of the sentences in the reference summaries

    Returns:
      A float: F_lcs

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    # total number of words in reference sentences
    m = len(_split_into_words(reference_sentences))

    # total number of words in evaluated sentences
    n = len(_split_into_words(evaluated_sentences))

    union_lcs_sum_across_all_references = 0
    for ref_s in reference_sentences:
        union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences,
                                                          ref_s)
    return _f_p_r_lcs(union_lcs_sum_across_all_references, m, n)


def rouge(hypotheses, references):
    """Calculates average rouge scores for a list of hypotheses and
    references"""

    # Filter out hyps that are of 0 length
    # hyps_and_refs = zip(hypotheses, references)
    # hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
    # hypotheses, references = zip(*hyps_and_refs)

    # Calculate ROUGE-1 F1, precision, recall scores
    rouge_1 = [
        rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)
    ]
    rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

    # Calculate ROUGE-2 F1, precision, recall scores
    rouge_2 = [
        rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)
    ]
    rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

    # Calculate ROUGE-L F1, precision, recall scores
    rouge_l = [
        rouge_l_sentence_level([hyp], [ref])
        for hyp, ref in zip(hypotheses, references)
    ]
    rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))

    return {
        #"rouge_1/f_score": rouge_1_f,
        "rouge_1/r_score": rouge_1_r,
        #"rouge_1/p_score": rouge_1_p,
        "rouge_2/f_score": rouge_2_f,
        "rouge_2/r_score": rouge_2_r,
        #"rouge_2/p_score": rouge_2_p,
        #"rouge_l/f_score": rouge_l_f,
        "rouge_l/r_score": rouge_l_r,
        #"rouge_l/p_score": rouge_l_p,
    }


def compute_bleu(target_corpus, predicted_corpus, max_order: int = 4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      target_corpus: list of references for each translation. Each
          reference should be tokenized into a list of tokens.
      predicted_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """

    # Include this version of get_ngrams, as it is different for bleu and rouge in chicken-sink code
    def _get_ngrams(segment, max_order):
        """Extracts all n-grams upto a given maximum order from an input segment.

        Args:
          segment: text segment from which n-grams will be extracted.
          max_order: maximum length in tokens of the n-grams returned by this
              methods.

        Returns:
          The Counter containing all n-grams up to max_order in segment
          with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i + order])
                ngram_counts[ngram] += 1
        return ngram_counts


    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (reference, translation) in zip(target_corpus,
                                        predicted_corpus):
        reference_length += len(reference)
        translation_length += len(translation)

        merged_ref_ngram_counts = _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0.] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return bleu, precisions, bp, ratio, translation_length, reference_length


def evaluate(pred_dir):
    predictions = {}
    for i in range(0, 1):
        #predictions['cnn_sec2ans_ext_{}'.format(i)] = "/data/saveryme/kitchen_sink/models/bart/bart_hf_predictions.csv"
        #predictions['t5_bioasq_sec2ans_ext_{}'.format(i)] = "{0}/t5-base_single_task_10_epochs/bioasq/single_doc/{1}/chiqa/section2answer_single_extractive/test/predictions.csv".format(base_dir, i)
        #predictions['t5_bioasq_sec2ans_abs_{}'.format(i)] = "{0}/t5-base_single_task_10_epochs/bioasq/single_doc/{1}/chiqa/section2answer_single_abstractive/test/predictions.csv".format(base_dir, i)
        #predictions['bart_bioasq_sec2ans_ext_{}'.format(i)] = "{0}/bart-large_single_task_10_epochs/bioasq/single_doc/{1}/chiqa/section2answer_single_extractive/test/predictions.csv".format(base_dir, i)
        predictions['bart_bioasq_sec2ans_abs_{}'.format(i)] = "{0}/bart-large_single_task_10_epochs/bioasq/single_doc/{1}/chiqa/section2answer_single_abstractive/test/predictions.csv".format(base_dir, i)
        #predictions['bioasq_sec2ans_ext_{}'.format(i)] = "{0}/bart-large-cnn_single_task_10_epochs/bioasq/single_doc/{1}/chiqa/section2answer_single_extractive/test/predictions.csv".format(base_dir, i)
        #predictions['chiqa_ext_sec2ans'] = "{}/t5-base_single_chiqa_10_epochs/chiqa/section2answer_single_extractive/chiqa/section2answer_single_extractive/test[80%:]/predictions.csv".format(base_dir)
        #predictions['chiqa_abs_sec2ans'] = "{}/t5-base_single_chiqa_10_epochs/chiqa/section2answer_single_abstractive/chiqa/section2answer_single_abstractive/test[80%:]/predictions.csv".format(base_dir)
        
    print(predictions)
    results = []
    extractive_results = []
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    #headers = ['Task', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU-4']
    extractive_headers = ["Task", "Sentences in Article", "Geo-mean"]
    headers = ['Task', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU-4']
    for task in predictions:
        print(task)
        prediction_list = []
        target_list = []
        prompts = []
        try:
            with open(predictions[task]) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    prediction_list.append(row['predictions'])
                    target_list.append(row['targets'])
                    prompts.append(row['prompt'])
        except FileNotFoundError as e:
            print(e)
            continue

        # Hack in bart predictions from earlier asumm work
        with open("/data/saveryme/asumm/asumm_data/predictions/bart_predictions/bart_chiqa_with_question_section2answer_singleAbstractive.json", "r", encoding="utf-8") as f: 
            predictions = json.load(f)
            prediction_list = []
            target_list = []
            prompts = []
            for question, ref, gen in zip(predictions['question'], predictions['ref_summary'], predictions['gen_summary']):
                prediction_list.append(gen)
                target_list.append(ref)
                prompts.append(question)

        print("Number of preds: ", len(prompts))
        targets = [*map(str.split, target_list)]
        split_predictions = [*map(str.split, prediction_list)]
        print(split_predictions[0:1])
        print(targets[0:1])
        # Compute the metrics for the predictions
        bleus = compute_bleu(target_corpus=targets, predicted_corpus=split_predictions)
        # Testing nltk tokenization
        split_predictions = [nltk.word_tokenize(summ) for summ in prediction_list]
        print(split_predictions[0:1])
        print(targets[0:1])
        targets = [nltk.word_tokenize(summ) for summ in target_list]
        nltk_targets = [[t] for t in targets]
        assert isinstance(nltk_targets, list)
        assert isinstance(nltk_targets[0], list)
        assert isinstance(nltk_targets[0][0], list)
        nltk_bleu = 100 * nltk.translate.bleu_score.corpus_bleu(nltk_targets, split_predictions)
        print("nltk BLEU: ", nltk_bleu)
        bleu_scores.append(bleus[0] * 100)
        rouges = rouge(references=targets, hypotheses=split_predictions)
        rouge1_scores.append(rouges['rouge_1/r_score'] * 100.)
        rouge2_scores.append(rouges['rouge_2/r_score'] * 100.)
        rougeL_scores.append(rouges['rouge_l/r_score'] * 100.)
        results.append([task,
                        rouges['rouge_1/r_score'] * 100.,
                        rouges['rouge_2/r_score'] * 100.,
                        rouges['rouge_l/r_score'] * 100.,
                        bleus[0] * 100.,
                        ])

        # Wilcoxon time
        if args.calculate_wilcoxon:
            score_dict = {
                'model_1': "list of models rouges per example", 
                'model_2': "list of model 2 rouges",
                }

        # Calculate the extractiveness of the generated summaries to the prompt
        if args.extractiveness:
            sent_match = 0
            total_sent = 0
            split_summs = [*map(lambda s: s.split("."), prediction_list)]
            split_articles = [*map(lambda s: s.split("."), prompts)]
            for summ, source in zip(split_summs, split_articles):
                for sentence in summ:
                    total_sent += 1
                    if sentence in source:
                        sent_match += 1
            print("Total number of sentences in summary:", total_sent)
            print("Total matches b/w summ and source", sent_match)
            sentence_ratio = sent_match / total_sent

            split_articles = [*map(lambda s: s.split(), prompts)]
            geo_mean = bleu_3_7(split_predictions, split_articles)
            #geo_mean = bleu_3_7(targets, split_articles)
            print("Geo mean is: ", geo_mean, "\n")
            extractive_results.append([task, sentence_ratio, geo_mean])

        # Write rouge for each prediction to excel
        if args.per_sample:
            per_sample_rouge2_scores = []
            per_sample_bleu_scores = []
            for ref, hyp in zip(targets, split_predictions):
                bleus = compute_bleu(target_corpus=[ref], predicted_corpus=[hyp])
                per_sample_bleu_scores.append(bleus[0] * 100)
                rouges = rouge(references=[ref], hypotheses=[hyp])
                per_sample_rouge2_scores.append(rouges['rouge_2/r_score'] * 100)
            ann_dict = {'Inputs': prompts, 'Targets': targets, 'Predictions': split_predictions, 'ROUGE-2': per_sample_rouge2_scores, 'BLEU': per_sample_bleu_scores}
            df = pd.DataFrame(ann_dict)
            df.to_excel("results/{}_metrics-per-summ.xlsx".format(task), index=False)

    print(tabulate(results, headers=headers))
    if args.extractiveness:
        print("\n", tabulate(extractive_results, headers=extractive_headers))

    if args.error_estimates and len(predictions) > 1:
        print(np.round(statistics.mean(rouge2_scores), 4))
        print(np.round(statistics.stdev(rouge2_scores), 4))
        print(np.round(statistics.mean(bleu_scores), 4))
        print(np.round(statistics.stdev(bleu_scores), 4))
        # Currently not using other rouge measures
        #print(np.round(statistics.mean(rouge1_scores), 4))
        #print(np.round(statistics.mean(rougel_scores), 4))
        #print(np.round(statistics.stdev(rouge1_scores), 4))
        #print(np.round(statistics.stdev(rougel_scores), 4))



if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    evaluate()
