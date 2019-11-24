import bisect
import gc
import glob
import random
import numpy as np

import torch

from others.logging import logger



class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def _pad_alignment(self, data, pad_number, eps=1e-8):
        width_tgt = max(len(d) for d in data)
        width_src = max(len(d[0]) for d in data)
        rtn_data = [];
        for ex in data:
            ex = np.array(ex)
            ex = np.add(ex, eps)
            norm = np.sum(ex, axis=1)
            ex = ex / norm[:, None]
            ex = ex.tolist()
            ex = [row + [pad_number] * (width_src - len(row)) for row in ex]
            for i in range(len(ex), width_tgt):
                ex.append([0] * width_src)
            rtn_data.append(ex)
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False, max_pos=-1):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            p_pair = [x[0] for x in data]
            n_pair = [x[1] for x in data]
            p_segs = [x[2] for x in data]
            n_segs = [x[3] for x in data]
            p_summ_mask = [x[4] for x in data]
            n_summ_mask = [x[5] for x in data]

            if max_pos > -1:
                width = max_pos
            else:
                width = -1
                
            p_pair = torch.tensor(self._pad(p_pair, 0, width))
            n_pair = torch.tensor(self._pad(n_pair, 0, width))

            p_segs = torch.tensor(self._pad(p_segs, 0, width))
            n_segs = torch.tensor(self._pad(n_segs, 0, width))

            p_summ_mask = torch.tensor(self._pad(p_summ_mask, 0, width))
            n_summ_mask = torch.tensor(self._pad(n_summ_mask, 0, width))

            p_summ_mask = (p_summ_mask == 1)
            n_summ_mask = (n_summ_mask == 1)

            p_mask = 1 - (p_pair == 0)
            n_mask = 1 - (n_pair == 0)

            setattr(self, 'p_pair', p_pair.to(device))
            setattr(self, 'n_pair', n_pair.to(device))
            setattr(self, 'p_segs', p_segs.to(device))
            setattr(self, 'n_segs', n_segs.to(device))
            setattr(self, 'p_mask', p_mask.to(device))
            setattr(self, 'n_mask', n_mask.to(device))
            setattr(self, 'p_summ_mask', p_summ_mask.to(device))
            setattr(self, 'n_summ_mask', n_summ_mask.to(device))

            if (is_test):
                src_str = [x[-3] for x in data]
                setattr(self, 'src_txt', src_str)
                tgt_str = [x[-2] for x in data]
                setattr(self, 'p_tgt_txt', tgt_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'n_tgt_txt', tgt_str)

    def __len__(self):
        return self.batch_size


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "dev", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    if (len(new) == 4):
        pass
    p_pair, n_pair = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, max(len(p_pair), len(n_pair)))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle
        self.max_pos = args.max_pos

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        if (self.args.task == 'abs'):
            self.batch_size_fn = abs_batch_size_fn
        else:
            self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        p_pair = ex['pos']
        n_pair = ex['neg']
        p_segs = ex['p_segs']
        n_segs = ex['n_segs']
        p_summ_mask = ex['p_summ_mask']
        n_summ_mask = ex['n_summ_mask']
        if(not self.args.use_interval):
            p_segs=[0]*len(p_segs)
        if(not self.args.use_interval):
            n_segs=[0]*len(n_segs)
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        n_tgt_txt = ex['n_tgt_text']

        if(is_test):
            return p_pair, n_pair, p_segs, n_segs, p_summ_mask, n_summ_mask, src_txt, tgt_txt, n_tgt_txt
        else:
            return p_pair, n_pair, p_segs, n_segs, p_summ_mask, n_summ_mask

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['pos'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))

            p_batch = self.batch(p_batch, self.batch_size)


            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test, self.max_pos)

                yield batch
            return


'''
class TextDataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.batch_size = batch_size
        self.device = device

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if (not self.args.use_interval):
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        alignment = ex['alignment']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id

        for i in range(len(alignment)):
            end_s = [alignment[i][-1]]
            alignment[i] = alignment[i][:-1][:self.args.max_pos - 1] + end_s

        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        if (is_test):
            return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels, alignment

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = batch(p_batch, self.batch_size)

            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if (len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return
'''
