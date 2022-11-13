import logging
import torch
import random
import numpy as np

from sklearn.metrics import f1_score, recall_score, precision_score
from preprocess.data_preprocess import get_data_loader, load_tag_2id, load_data_from_brat
from model.tokenizer import Tokenizer
from model.info_extract import InfoModel, InfoModelConfig
from model.conv_attention import ConvAttentionConfig
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 1
BATCH_SIZE = 16
EPOCH = 40
VALID_SET_PATIO = 0.1
DEVICE = 'cuda'

TRAIN_PATH = './data/train/'
VAL_PATH = './data/val/'
TOTAL_PATH = './data/total/'

CHECK_POINT_PATH = './ckpt/'

ENCODER = 'roberta'
HIDDEN_SIZE = 312
MAX_LENGTH = 512
ENCODER_LR = 5e-5
VOCAB_PATH = './pretrained/roberta/vocab.txt'
PRETRAINED = './pretrained/roberta/'

PAIRIds = {
    1: [3, 5, 9],
    7: [1]
}

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


def train(train_loader, val_loader, entity_nums, relation_nums):
    ner_labels = np.arange(0, entity_nums)
    relation_labels = np.arange(0, relation_nums)
    conv_config = ConvAttentionConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=HIDDEN_SIZE,
        kernel_size=5,
        num_hidden_layers=6,
        max_length=MAX_LENGTH,
        num_attention_heads=4
    )

    config = InfoModelConfig(
        hidden_size=HIDDEN_SIZE,
        entity_nums=entity_nums,
        relation_nums=relation_nums,
        pretrained_path=PRETRAINED
    )
    model = InfoModel(config, conv_config, encoder_name=ENCODER, pairIds=PAIRIds)
    # model.load_state_dict(torch.load(CHECK_POINT_PATH + ENCODER + '_info_model_1.pth', map_location='cpu'))
    model.to(DEVICE)

    lr = 1e-3
    max_score = 0
    logger.info('start training')

    freeze = False
    for e in range(EPOCH):
        model.train()
        base_params = list(map(id, model.encoder.parameters()))
        other_params = filter(lambda p: id(p) not in base_params, model.parameters())
        params = [
            {'params': other_params, 'lr': lr},
            {'params': model.encoder.parameters(), 'lr': min(lr, ENCODER_LR)}
        ]
        optimizer = torch.optim.Adam(params, lr=lr)
        for i, (x_batch, mask_batch, ner_batch, relation_batch, relation_mask_batch) in enumerate(train_loader):
            loss, rl_loss = model.joint_loss(x_batch.to(DEVICE),
                                             mask_batch.to(DEVICE),
                                             ner_batch.to(DEVICE),
                                             relation_batch.to(DEVICE),
                                             relation_mask_batch.to(DEVICE))
            # 损失函数的两种方式
            # if e>5 and loss.item() > rl_loss.item():
            #     rl_loss *= (loss.item()/rl_loss.item())
            # loss += rl_loss

            if loss.item() < rl_loss.item():
                rl_loss *= (loss.item() / (rl_loss.item() if rl_loss.item() > 0 else 1))
            if e > 5:
                loss += rl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), CHECK_POINT_PATH + f'{ENCODER}_info_model.pth')

        model.eval()
        pred_ner = []
        pred_relation = []
        valid_ner = []
        valid_relation = []
        for i, (x_batch, mask_batch, ner_batch, relation_batch, relation_mask_batch) in enumerate(val_loader):
            with torch.no_grad():
                ner_logits, relation_logits, relation_mask = model(x_batch.to(DEVICE),
                                                                   mask_batch.to(DEVICE),
                                                                   relation_mask_batch.to(DEVICE))
                ner_logits = ner_logits.argmax(-1).view(-1).cpu().data.numpy().tolist()
                relation_pred = relation_logits.argmax(-1).view(-1)

                span_mask = relation_mask.view(size=(-1,))
                relation_pred = relation_pred[span_mask > 0]
                relation_pred = relation_pred.cpu().data.numpy().tolist()
                relation_real = relation_batch.view(-1)
                relation_real = relation_real[span_mask > 0]
                relation_real = relation_real.numpy().tolist()

            pred_ner += ner_logits
            pred_relation += relation_pred
            valid_ner += ner_batch.view(-1).numpy().tolist()
            valid_relation += relation_real

        precision = precision_score(np.array(pred_ner), np.array(valid_ner), average='macro')
        recall = recall_score(np.array(pred_ner), np.array(valid_ner), average='macro')
        ner_f1_score = f1_score(np.array(pred_ner), np.array(valid_ner), average='macro', labels=ner_labels)

        relation_precision = precision_score(np.array(pred_relation), np.array(valid_relation), average='macro')
        relation_recall = recall_score(np.array(pred_relation), np.array(valid_relation), average='macro')
        relation_f1_score = f1_score(np.array(pred_relation), np.array(valid_relation), average='macro',
                                     labels=relation_labels)

        score = ner_f1_score
        if not freeze:
            score += relation_f1_score

        print('*' * 100)
        print('实体准确率为：', precision, '实体召回率为：', recall, '实体f1得分为：', ner_f1_score)
        print('关系准确率为：', relation_precision, '关系召回率为：', relation_recall, '关系f1得分为：', relation_f1_score)

        if score > max_score:
            max_score = score
            lr *= 0.5
        else:
            lr *= 0.1
        lr = max(2e-5, lr)


if __name__ == '__main__':
    tokenizer = Tokenizer()
    tokenizer.load_vocab(VOCAB_PATH)

    entity2id, relation2id, relation_pairs = load_tag_2id(TRAIN_PATH)

    entity_nums = len(entity2id) - 1
    relation_nums = len(relation2id)

    print(entity2id)
    print(relation2id)
    print(relation_pairs)

    # 训练测试数据划分方式一
    # train_data, _, _, _ = load_data_from_brat(TRAIN_PATH)
    # val_data, _, _, _ = load_data_from_brat(VAL_PATH)
    # train_loader = get_data_loader(train_data, tokenizer, entity2id, relation2id, relation_pairs, MAX_LENGTH)
    # val_loader = get_data_loader(val_data, tokenizer, entity2id, relation2id, relation_pairs, MAX_LENGTH)
    # train(train_loader, val_loader, entity_nums, relation_nums)

    # 训练测试数据划分方式二
    total_data, _, _, _ = load_data_from_brat(TOTAL_PATH)
    random.shuffle(total_data)
    total_size = len(total_data)
    split_index = int(0.8*total_size)
    train_data = total_data
    val_data = total_data[split_index:]

    train_loader = get_data_loader(train_data, tokenizer, entity2id, relation2id, relation_pairs, MAX_LENGTH)
    val_loader = get_data_loader(val_data, tokenizer, entity2id, relation2id, relation_pairs, MAX_LENGTH)
    train(train_loader, val_loader, entity_nums, relation_nums)
