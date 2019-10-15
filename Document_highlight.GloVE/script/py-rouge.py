import rouge


def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


for aggregator in ['Avg', 'Best', 'Individual']:
    print('Evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = None
    #apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)


    hypothesis_1 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia 's top political leaders , saying the meeting would not bring any progress in deadlocked negotiations to form a government .\nGovernment and opposition parties have asked King Norodom Sihanouk to host a summit meeting after a series of post-election negotiations between the two opposition groups and Hun Sen 's party to form a new government failed .\nHun Sen 's ruling party narrowly won a majority in elections in July , but the opposition _ claiming widespread intimidation and fraud _ has denied Hun Sen the two-thirds vote in parliament required to approve the next government .\n"
    references_1 = ["Prospects were dim for resolution of the political crisis in Cambodia in October 1998.\nPrime Minister Hun Sen insisted that talks take place in Cambodia while opposition leaders Ranariddh and Sam Rainsy, fearing arrest at home, wanted them abroad.\nKing Sihanouk declined to chair talks in either place.\nA U.S. House resolution criticized Hun Sen's regime while the opposition tried to cut off his access to loans.\nBut in November the King announced a coalition government with Hun Sen heading the executive and Ranariddh leading the parliament.\nLeft out, Sam Rainsy sought the King's assurance of Hun Sen's promise of safety and freedom for all politicians."]

    all_hypothesis = [hypothesis_1]
    all_references = [references_1]

    scores = evaluator.get_scores(all_hypothesis, all_references)

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
            print()
        else:
            print(prepare_results(results['p'], results['r'], results['f']))
            print (metric)
            print (results['p'] * 100)
            print (results['r'] * 100)
            print (results['f'] * 100)
    print()
