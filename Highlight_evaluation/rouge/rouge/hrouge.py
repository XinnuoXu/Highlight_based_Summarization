# -*- coding: utf-8 -*-
from __future__ import absolute_import
import six
#import rouge.rouge_score as rouge_score
import hrouge_score as rouge_score
import io
import os


class FilesHRouge:
    def __init__(self, *args, **kwargs):
        """See the `HRouge` class for args
        """
        self.rouge = HRouge(*args, **kwargs)

    def _check_files(self, hyp_path, ref_path):
        assert(os.path.isfile(hyp_path))
        assert(os.path.isfile(ref_path))

        def line_count(path):
            count = 0
            with open(path, "rb") as f:
                for line in f:
                    count += 1
            return count

        hyp_lc = line_count(hyp_path)
        ref_lc = line_count(ref_path)
        assert(hyp_lc == ref_lc)

    def get_scores(self, hyps, refs, ref_weights, avg=False, ignore_empty=False):
        """Calculate ROUGE scores between each pair of
        lines (hyp_file[i], ref_file[i]).
        Args:
          * hyp_path: hypothesis file path
          * ref_path: references file path
          * avg (False): whether to get an average scores or a list
        """
        return self.rouge.get_scores(hyps[:-1], refs[:-1], ref_weights[:-1], avg=avg, ignore_empty=ignore_empty)


class HRouge:
    DEFAULT_METRICS = ["rouge-1", "rouge-2"]
    AVAILABLE_METRICS = {
        "rouge-1": lambda hyp, ref, ref_weight: rouge_score.rouge_n(hyp, ref, ref_weight, 1),
        "rouge-2": lambda hyp, ref, ref_weight: rouge_score.rouge_n(hyp, ref, ref_weight, 2),
        "rouge-l": lambda hyp, ref:
            rouge_score.rouge_l_summary_level(hyp, ref),
    }
    DEFAULT_STATS = ["f", "p", "r"]
    AVAILABLE_STATS = ["f", "p", "r"]

    def __init__(self, metrics=None, stats=None, return_lengths=False):
        self.return_lengths = return_lengths
        if metrics is not None:
            self.metrics = [m.lower() for m in metrics]

            for m in self.metrics:
                if m not in HRouge.AVAILABLE_METRICS:
                    raise ValueError("Unknown metric '%s'" % m)
        else:
            self.metrics = HRouge.DEFAULT_METRICS

        if stats is not None:
            self.stats = [s.lower() for s in stats]

            for s in self.stats:
                if s not in HRouge.AVAILABLE_STATS:
                    raise ValueError("Unknown stat '%s'" % s)
        else:
            self.stats = HRouge.DEFAULT_STATS

    def get_scores(self, hyps, refs, ref_weights, avg=False, ignore_empty=False):
        if isinstance(hyps, six.string_types):
            hyps, refs = [hyps], [refs]

        if ignore_empty:
            # Filter out hyps of 0 length
            hyps_and_refs = zip(hyps, refs)
            hyps_and_refs = [_ for _ in hyps_and_refs
                             if len(_[0]) > 0
                             and len(_[1]) > 0]
            hyps, refs = zip(*hyps_and_refs)

        assert(type(hyps) == type(refs))
        assert(len(hyps) == len(refs))

        if not avg:
            return self._get_scores(hyps, refs, ref_weights)
        return self._get_avg_scores(hyps, refs, ref_weights)

    def _get_scores(self, hyps, refs, ref_weights):
        scores = []
        for hyp, ref in zip(hyps, refs, ref_weights,):
            sen_score = {}

            hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]

            for m in self.metrics:
                fn = HRouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref)
                sen_score[m] = {s: sc[s] for s in self.stats}

            if self.return_lengths:
                lengths = {
                    "hyp": len(" ".join(hyp).split()),
                    "ref": len(" ".join(ref).split())
                }
                sen_score["lengths"] = lengths
            scores.append(sen_score)
        return scores

    def _get_avg_scores(self, hyps, refs, ref_weights):
        scores = {m: {s: 0 for s in self.stats} for m in self.metrics}
        if self.return_lengths:
            scores["lengths"] = {"hyp": 0, "ref": 0}

        count = 0
        for (hyp, ref, ref_weight) in zip(hyps, refs, ref_weights):
            hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]

            tmp_ref = []; tmp_weight = []; sen_ref = []; sen_weight = []
            for i, tok in enumerate(ref.split(" ")):
                if tok == ".":
                    tmp_ref.append(" ".join(sen_ref))
                    tmp_weight.append(sen_weight)
                    sen_ref = []
                    sen_weight = []
                else:
                    sen_ref.append(tok)
                    sen_weight.append(ref_weight[i])
            if len(sen_ref) > 0:
                tmp_ref.append(" ".join(sen_ref))
                tmp_weight.append(sen_weight)
            ref = tmp_ref; ref_weight = tmp_weight

            for m in self.metrics:
                fn = HRouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref, ref_weight)
                scores[m] = {s: scores[m][s] + sc[s] for s in self.stats}

            if self.return_lengths:
                scores["lengths"]["hyp"] += len(" ".join(hyp).split())
                scores["lengths"]["ref"] += len(" ".join(ref).split())

            count += 1
        avg_scores = {
            m: {s: scores[m][s] / count for s in self.stats}
            for m in self.metrics
        }

        if self.return_lengths:
            avg_scores["lengths"] = {
                k: scores["lengths"][k] / count
                for k in ["hyp", "ref"]
            }

        return avg_scores
