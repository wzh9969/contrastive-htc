from transformers import AutoTokenizer
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import json
from collections import defaultdict

np.random.seed(7)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    source = []
    labels = []
    label_ids = []
    label_dict = {}
    hiera = defaultdict(set)
    with open('wos_total.json', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            source.append(tokenizer.encode(line['doc_token'].strip().lower(), truncation=True))
            labels.append(line['doc_label'])
    for l in labels:
        if l[0] not in label_dict:
            label_dict[l[0]] = len(label_dict)
    for l in labels:
        assert len(l) == 2
        if l[1] not in label_dict:
            label_dict[l[1]] = len(label_dict)
        label_ids.append([label_dict[l[0]], label_dict[l[1]]])
        hiera[label_ids[-1][0]].add(label_ids[-1][1])
    value_dict = {i: tokenizer.encode(v.lower(), add_special_tokens=False) for v, i in label_dict.items()}
    torch.save(value_dict, 'bert_value_dict.pt')

    with open('tok.txt', 'w') as f:
        for s in source:
            f.writelines(' '.join(map(lambda x: str(x), s)) + '\n')
    with open('Y.txt', 'w') as f:
        for s in label_ids:
            one_hot = [0] * len(label_dict)
            for i in s:
                one_hot[i] = 1
            f.writelines(' '.join(map(lambda x: str(x), one_hot)) + '\n')

    from fairseq.binarizer import Binarizer
    from fairseq.data import indexed_dataset

    for data_path in ['tok', 'Y']:
        offsets = Binarizer.find_offsets(data_path + '.txt', 1)
        ds = indexed_dataset.make_builder(
            data_path + '.bin',
            impl='mmap',
            vocab_size=tokenizer.vocab_size,
        )
        Binarizer.binarize(
            data_path + '.txt', None, lambda t: ds.add_item(t), offset=0, end=offsets[1], already_numberized=True,
            append_eos=False
        )
        ds.finalize(data_path + '.idx')

    id = [i for i in range(len(source))]
    np_data = np.array(id)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    train = list(train)
    val = list(val)
    test = list(test)
    torch.save({'train': train, 'val': val, 'test': test}, 'split.pt')
