from info_extractor import EventExtractor
from model.info_extract import InfoModel, InfoModelConfig
import torch
import os
import time

CHECK_POINT_PATH = './ckpt/'
ENCODER = 'roberta'
HIDDEN_SIZE = 312
MAX_LENGTH = 512
VOCAB_PATH = './pretrained/roberta/vocab.txt'
PRETRAINED = './pretrained/roberta/'

e2id = {'none': 0, 'pad': -1, '组织-B': 1, '组织-I': 2, '职位-B': 3, '职位-I': 4, '人物-B': 5, '人物-I': 6, '违法违纪事件-B': 7, '违法违纪事件-I': 8, '处理措施-B': 9, '处理措施-I': 10}
r2id = {'none': 0, '处理结果': 1, '所在组织': 2, '所在职位': 3, '违法人员': 4}
id2r = {r2id[name]: name for name in r2id}
pairs = {'所在职位': '人物_职位', '违法人员': '人物_违法违纪事件', '处理结果': '处理措施_人物', '所在组织': '人物_组织'}
rid2pid = {}

for relation in r2id:
    rid = r2id[relation]
    if rid == 0:
        continue
    pair = pairs[relation]
    start, end = pair.split('_')[0], pair.split('_')[1]
    start_id = e2id[start + '-B']
    end_id = e2id[end + '-B']
    rid2pid[rid] = [start_id, end_id]
print(rid2pid)

if __name__ == '__main__':
    config = InfoModelConfig(
        hidden_size=HIDDEN_SIZE,
        entity_nums=len(e2id) - 1,
        relation_nums=len(r2id),
        pretrained_path=PRETRAINED
    )
    models = []
    name = 'roberta_info_model.pth'
    model = InfoModel(config, encoder_name=ENCODER)
    model.load_state_dict(torch.load(CHECK_POINT_PATH + name, map_location='cpu'))
    model.eval()
    extractor = EventExtractor(HIDDEN_SIZE,
                               e2id,
                               VOCAB_PATH,
                               model,
                               id2r,
                               rid2pid,
                               max_length=MAX_LENGTH
                               )

    with open('10.txt', 'r', encoding='utf-8') as f:
        content = ''.join(f.readlines())
    start = time.time()
    print('content:', content)
    entities, relations = extractor.extract(content)

    print('耗时时间为：', time.time() - start)
    for item in entities:
        print('实体为：', item)

    for item in relations:
        from_idx, to_idx, name = item[0], item[1], item[2]
        print('关系为：', entities[from_idx], entities[to_idx], name)
