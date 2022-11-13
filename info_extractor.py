import torch
from model.tokenizer import Tokenizer


class EventExtractor():
    def __init__(self,
                 hidden_size,
                 tag2id,
                 vocab_path,
                 models,
                 id2relation,
                 rid2pid,
                 max_length=512,
                 device='cpu'):
        self.hidden_size = hidden_size
        self.id2tag = {tag2id[tag]: tag for tag in tag2id}
        self.device = device
        self.models = models
        self.id2relation = id2relation
        self.rid2pid = rid2pid
        self.max_length = max_length

        self.load_tokenizer(vocab_path)

    def load_tokenizer(self, path):
        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocab(path)

    def ner_decode(self, text, tag_seq):
        entities = []
        pos2ent = {}
        trigger = False
        name = ''
        start = 0
        for i in range(len(tag_seq)):
            tag = self.id2tag[tag_seq[i]]
            if trigger:
                if tag != name + '-I':
                    value = text[start:i]
                    entities.append([value, name, start, i])
                    pos2ent[start] = len(entities) - 1
                    if '-B' not in tag:
                        trigger = False
                    else:
                        trigger = True
                        name = tag[:-2]
                        start = i
            else:
                if '-B' in tag:
                    trigger = True
                    name = tag[:-2]
                    start = i
        return entities, pos2ent

    def relation_decode(self, pos2ent, relation_pairs):
        relations = []
        for pair in relation_pairs:
            from_idx, to_idx, relname = pair[0], pair[1], pair[2]
            relations.append([pos2ent[from_idx], pos2ent[to_idx], relname])
        return relations

    def get_relation_pairs(self, shift, ner_seq, relation_matrix):
        relation_pairs = []
        for i in range(len(relation_matrix)):
            for j in range(len(relation_matrix)):
                relid = relation_matrix[i][j]
                start_ent_tag = ner_seq[i]
                end_ent_tag = ner_seq[j]
                if relid in self.rid2pid:
                    relation_pair_ids = self.rid2pid[relid]
                    if start_ent_tag == relation_pair_ids[0] and end_ent_tag == relation_pair_ids[1]:
                        relation_pairs.append([i+shift, j+shift, self.id2relation[relid]])
        return relation_pairs

    def predict(self, text):
        input_x, attention_mask = self.tokenizer.convert_to_ids(text, max_length=self.max_length, attention_mask=True, auto=True)
        with torch.no_grad():
            input_x = torch.tensor(input_x).long().unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(attention_mask).long().unsqueeze(0).to(self.device)
            tag_seq = []
            relation_matrix = []
            tag_seq_, relation_matrix_, _ = self.models(input_x, attention_mask)
            print('tag_seq_:', tag_seq_.shape)
            tag_seq.append(tag_seq_)
            relation_matrix.append(relation_matrix_)
            tag_seq = torch.cat(tag_seq, 0)
            relation_matrix = torch.cat(relation_matrix, 0)
            tag_seq = tag_seq.mean(0)
            relation_matrix = relation_matrix.mean(0)

            tag_seq = tag_seq.argmax(-1).cpu().data.numpy().tolist()[1:-1]
            relation_matrix = relation_matrix.argmax(-1)[1:-1, 1:-1].cpu().data.numpy()
        return tag_seq, relation_matrix

    def extract(self, text):
        tag_seq = []
        relation_pairs = []
        index = 0
        while index < len(text):
            span = text[index: index+self.max_length]
            span_ner_seq, span_relation_seq = self.predict(span)

            span_relation_pairs = self.get_relation_pairs(index, span_ner_seq, span_relation_seq)
            tag_seq += span_ner_seq
            relation_pairs += span_relation_pairs

            index += self.max_length - 2

        entities, pos2ent = self.ner_decode(text, tag_seq)
        relations = self.relation_decode(pos2ent, relation_pairs)
        return entities, relations