import os
import json
import jsonlines
import numpy as np

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if tokenizer == 'bert' or 'clip' in tokenizer:
                filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, '%s_%s_enc_xlmr.json' % (dataset.upper(), split))
            else:
                raise NotImplementedError('unspported tokenizer %s' % tokenizer)
            # json file is a python list
            with open(filepath) as f:
                new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

            if not is_test:
                if dataset == 'r4r' and split == 'val_unseen':
                    ridxs = np.random.permutation(len(new_data))[:200]
                    new_data = [new_data[ridx] for ridx in ridxs]
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            filepath = os.path.join(anno_dir, split)
            # convert jsonl to python list
            new_data = []
            with jsonlines.open(filepath, 'r') as f:
                for item in f:
                    new_data.append(item)
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True):
    if "/" not in splits[0]: # the official splits
        data = []
        for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr
                if 'clip' in tokenizer:
                    from my_clip import clip as my_clip
                    new_item['instr_clip_encoding'], new_item['instr_clip_length'] = my_clip.tokenize(instr, truncate=True)
                    if 'bert' in tokenizer:
                        new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                else:
                    new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                    new_item['instr_clip_encoding'], new_item['instr_clip_length'] = None, None
                del new_item['instructions']
                del new_item['instr_encodings']
                data.append(new_item)
    else: # augmented data
        data = []
        for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer)):
            new_item = dict(item)
            new_item['path_id'] = item['instr_id'].split('_')[1]
            new_item['instr_id'] = item['instr_id']
            new_item['objId'] = None
            new_item['instruction'] = None
            new_item['instr_encoding'] = item['instr_encoding'][:max_instr_len]
            new_item['instr_clip_encoding'], new_item['instr_clip_length'] = None, None
            data.append(new_item)
            
    return data