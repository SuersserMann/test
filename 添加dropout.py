import re
import unicodedata
import numpy as np
import torch
from TorchCRF import CRF
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from transformers import AutoModel, AutoTokenizer
import json
import torch.utils.data as data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import torch.nn.functional as F
import jsonlines
import regex
import random

warnings.filterwarnings("ignore")

print(torch.__version__)

# device_ids = [0, 1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = AutoModel.from_pretrained("bert-base-chinese")
        self.pretrained.to(device)
        for param in self.pretrained.parameters():
            param.requires_grad_(False)
        self.lstm = nn.LSTM(768, 384, num_layers=2, batch_first=True, bidirectional=True)
        self.cc = nn.Linear(768, 208)
        self.dropout = nn.Dropout(0.5)  # 添加dropout层
        self.crf = CRF(208)  # 添加CRF层

    def forward(self, input_ids, attention_mask):
        out = self.pretrained(input_ids=input_ids, attention_mask=attention_mask)
        out = out.last_hidden_state
        out, _ = self.lstm(out)
        out = self.dropout(out)  # 应用dropout层
        out = self.cc(out)
        return out


model = Model()
# model = nn.DataParallel(model, device_ids=device_ids)
model.to(device)


class Dataset(data.Dataset):
    def __init__(self, filename):
        with jsonlines.open(filename, 'r') as f:
            self.data = list(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        text_id = item['text_id']
        text = item['text']
        events = item['events']

        # 过滤掉events为空的数据
        if events == []:
            return None

        return text_id, text, events


class Dataset1(data.Dataset):
    def __init__(self, filename):
        with jsonlines.open(filename, 'r') as f:
            self.data = list(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        text_id = item['text_id']
        text = item['text']
        events = item['events']

        return text_id, text, events


b_train_dataset = Dataset1('123/random_train.jsonl')
b_val_dataset = Dataset1('123/random_val.jsonl')
train_dataset = Dataset('123/train.jsonl')
val_dataset = Dataset('123/dev.jsonl')
da_train = Dataset('123/da_train.jsonl')
# test_dataset = Dataset('testA.jsonl')
b_train_dataset = [item for item in b_train_dataset]
b_val_dataset = [item for item in b_val_dataset]
da_train=[item for item in da_train]
train_dataset = [item for item in train_dataset if item is not None]
val_dataset = [item for item in val_dataset if item is not None]

entity_counts = {}
filtered_train_dataset = []

for text_id, text, events in train_dataset:
    entity_dict = {}
    for event in events:
        entity = event['entity']
        event_type = event['type']
        if entity in entity_dict:
            entity_dict[entity].add(event_type)
        else:
            entity_dict[entity] = {event_type}

    has_different_event_types = False
    for entity, event_types in entity_dict.items():
        num_event_types = len(event_types)
        if num_event_types > 1:
            has_different_event_types = True
            if num_event_types in entity_counts:
                entity_counts[num_event_types] += 1
            else:
                entity_counts[num_event_types] = 1

    if has_different_event_types:
        filtered_train_dataset.append((text_id, text, events))

train_dataset = [x for x in train_dataset if x not in filtered_train_dataset]

entity_counts_val = {}
filtered_val_dataset = []
for text_id, text, events in val_dataset:
    entity_dict = {}
    for event in events:
        entity = event['entity']
        event_type = event['type']
        if entity in entity_dict:
            entity_dict[entity].add(event_type)
        else:
            entity_dict[entity] = {event_type}

    has_different_event_types = False
    for entity, event_types in entity_dict.items():
        num_event_types = len(event_types)
        if num_event_types > 1:
            has_different_event_types = True
            if num_event_types in entity_counts_val:
                entity_counts_val[num_event_types] += 1
            else:
                entity_counts_val[num_event_types] = 1

    if has_different_event_types:
        filtered_val_dataset.append((text_id, text, events))
val_dataset = [x for x in val_dataset if x not in filtered_val_dataset]


def count_entity_occurrences(text, entity):
    count = 0
    start_index = 0
    while True:
        index = text.find(entity, start_index)
        if index == -1:
            break
        count += 1
        start_index = index + 1
    return count


def filter_dataset(dataset):
    filtered_dataset = []
    for text_id, text, events in dataset:
        entities = set(event['entity'] for event in events)
        for entity in entities:
            entity_count = count_entity_occurrences(text, entity)
            if entity_count == 2:
                filtered_dataset.append((text_id, text, events))
                break
    return filtered_dataset


quchong_train_dataset = filter_dataset(train_dataset)
quchong_val_dataset = filter_dataset(val_dataset)
train_dataset = [x for x in train_dataset if x not in quchong_train_dataset]
val_dataset = [x for x in val_dataset if x not in quchong_val_dataset]

# def filter_empty_events(dd):
#     add_0_dataset = [item for item in dd if item[2] == []]
#     return add_0_dataset
#
#
# train_0_dataset = filter_empty_events(b_train_dataset)
# val_0_dataset = filter_empty_events(b_val_dataset)
# random_train = random.sample(train_0_dataset, 10000)
# random_val = random.sample(val_0_dataset, 1000)
train_dataset.extend(b_train_dataset)
train_dataset.extend(da_train)
val_dataset.extend(b_val_dataset)

token = AutoTokenizer.from_pretrained("bert-base-chinese")

type_list = ['公司注销', '高层涉嫌违法', '高层变更', '关闭分支机构', '吊销资质牌照', '经营期限到期', '造假欺诈',
             '偷税漏税', '信息泄露', '重大赔付', '窃取别人商业机密', '违规催收', '网站安全漏洞', '实际控制人涉诉仲裁',
             '实际控制人违规', '实际控制人变更', '盗取隐私信息', '破产清算', '警告', '重大资产损失', '财务信息造假',
             '实际控制人涉嫌违法', '债务违约', '欠薪', '外部信用评级下调', '内幕交易', '偿付能力不足',
             '评级机构中止评级', '公司停牌', '债务融资失败', '资金紧张', '债务展期', '债务重组', '资产质量下降',
             '资本充足不足', '重大债务到期', '被银行停贷', '停产停业', '盈利能力下降', '高层失联/死亡', '延期信息披露',
             '资产冻结', '经营亏损', '投资亏损', '裁员', '股权冻结/强制转让', '澄清辟谣', '退出市场', '公司退市',
             '吊销业务许可或执照', '业务/资产重组', '连续下跌', '实际控制人失联/死亡', '企业被问询约谈审查', '挤兑',
             '员工罢工示威', '发放贷款出现坏账', '被接管', '股东利益斗争', '监管入驻', '产品违约/不足额兑付',
             '基层员工流失', '更换基金经理', '第一大股东变化', '履行连带担保责任', '重大安全事故', '经营激进',
             '无法表示意见', '股票发行失败', '保留意见', '出具虚假证明', '暂停上市', '责令改正', '禁入行业',
             '限制业务范围', '停止接受新业务', '产品虚假宣传', '公司违规关联交易', '非法集资', '扰乱市场秩序',
             '终身禁入行业', '撤销任职资格', '税务非正常户', '大量投诉', '总部被警方调查', '被列为失信被执行人',
             '分支机构被警方调查', '骗保', '监管评级下调', '财务报表更正', '否定意见', '自然灾害', '行业排名下降',
             '限制股东权利', '股权查封', '签订对赌协议', '审计师辞任', '股权融资失败', '停止批准增设分支机构',
             '薪酬福利下降', '误操作', '授信额度减少', '经营资质瑕疵']


def remove_unrecognized_unicode(text):
    cleaned_text = regex.sub(r'\p{C}', '', text)
    return cleaned_text


def find_indices(cfn_spans_start, word_start):
    matches = []
    for i, num in enumerate(word_start):
        if num in cfn_spans_start:
            matches.append(i)
    return matches


def reshape_and_remove_pad(outs, labels, attention_mask):
    outs = outs[attention_mask == 1]

    # Reshape 'labels' tensor based on attention_mask
    labels = labels[attention_mask == 1]

    return outs, labels


def get_index(target_type):
    return type_list.index(target_type)


def find_all(text, sub):
    start = 0
    while start < len(text):
        start = text.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


def collate_fn(data):
    text_id = []
    text = []
    events = []
    labels = []
    characters_list = []
    for i in data:
        # text_id.append(i[0])
        text_one = i[1]

        event_one = i[2]
        # events.append(i[2])
        text_one = re.sub(r'[ \u3000\xa0\u2002\u2003�]+', '', text_one)
        text_one = remove_unrecognized_unicode(text_one)
        # text.append(text_one print)
        characters = [char for char in text_one]
        # 恶心死我了
        if len(text_one) > 510:
            text_one = text_one[:510]
        len_text = len(text_one)

        label = [206] * len_text

        characters_list.append(characters)
        if event_one:
            for g, item in enumerate(event_one):
                entity_start = list(find_all(text_one, item['entity']))  # [8,12]
                type2id = get_index(item['type'])  # 13
                for t in range(len(entity_start)):
                    count_a = 0
                    for z in range(entity_start[t], len(item['entity']) + entity_start[t]):
                        if count_a == 0:
                            label[z] = type2id + 103
                            count_a += 1
                        else:
                            label[z] = type2id
        labels.append(label)

    data = token.batch_encode_plus(

        characters_list,
        padding=True,
        truncation=True,
        # max_length=512,
        return_tensors='pt',
        is_split_into_words=True,
        return_length=True)

    lens = data['input_ids'].shape[1]

    for i in range(len(labels)):
        # x = len(labels[i])
        # bb = text[i]
        labels[i] = [207] + labels[i]
        labels[i] += [207] * lens
        labels[i] = labels[i][:lens]
        # cc = characters_list[i]
        # if labels[i][0] != 207:
        #     print(f"在序列{i}的开头没有加[207]")
        # if labels[i][x + 1] != 207:
        #     print(f"在序列{i}的结尾没有加[207]")

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    # labels_tensor = torch.zeros(len(labels), lens, 208).to(device)
    # for i, label_seq in enumerate(labels):
    #     for j, label in enumerate(label_seq):
    #         labels_tensor[i][j][label] = 1
    labels = torch.LongTensor(labels).to(device)

    return input_ids, attention_mask, labels


# batchsize不能太大，明白了，数据太少了，刚才的数据被drop_last丢掉了
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=8,
                                           collate_fn=collate_fn,
                                           shuffle=True,
                                           drop_last=False)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=8,
                                         collate_fn=collate_fn,
                                         shuffle=True,
                                         drop_last=False)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=64,
#                                           collate_fn=collate_fn,
#                                           shuffle=False,
#                                           drop_last=False)


def train_model(learning_rate, num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 使用传入的学习率

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_val_f1 = 0
    best_model_state = None
    best_val_loss = 100
    patience = 5
    counter = 0
    threshold = 0.5
    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_f1 = 0
            train_acc = 0
            train_precision = 0
            train_recall = 0
            train_count = 0
            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):

                out = model(input_ids=input_ids, attention_mask=attention_mask)

                # out_z, labels_z = reshape_and_remove_pad(out, labels, attention_mask)

                loss = -model.module.crf(out, labels, attention_mask.to(torch.bool))
                loss = loss.sum() / train_loader.batch_size
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 梯度清零，防止梯度累积

                out = model.module.crf.viterbi_decode(out, attention_mask.to(torch.bool))
                # out = torch.argmax(out, dim=2)
                # out = torch.where(out > threshold, torch.ones_like(out), torch.zeros_like(out))

                true_labels = []

                list5 = []
                list6 = []
                predicted_labels = out
                for j in range(len(labels)):
                    true_label = labels[j].tolist()
                    t_first_index = true_label.index(207)
                    t_second_index = true_label.index(207, t_first_index + 1)
                    t_modified_label = true_label[t_first_index:t_second_index + 1]
                    true_labels.append(t_modified_label)

                for z, j in zip(predicted_labels, true_labels):
                    list3 = []
                    list4 = []
                    for m, n in zip(z, j):
                        if m != 206 or n != 206:
                            list3.append(m)
                            list4.append(n)
                    list5.append(list3)
                    list6.append(list4)

                y_true = [label for sublist in list6 for label in sublist]
                y_pred = [label for sublist in list5 for label in sublist]

                # accuracy = accuracy_score(true_labels, predicted_labels)

                precision = precision_score(y_true, y_pred, average='micro')
                recall = recall_score(y_true, y_pred, average='micro')
                f1 = f1_score(y_true, y_pred, average='micro')

                train_loss += loss.item()
                train_f1 += f1
                train_precision += precision
                train_recall += recall
                train_count += 1
                # print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                print(
                    f"第{epoch + 1}周期：第{i + 1}轮训练, loss：{loss.item()}, F1:{f1},precision:{precision},recall:{recall}")

            train_loss /= train_count
            train_f1 /= train_count
            train_precision /= train_count
            train_recall /= train_count

            print(
                f"----------总第{epoch + 1}周期,loss为{train_loss},训练集F1为{train_f1},precision为{train_precision}，recall为{train_recall}------------")

            # 验证
            model.eval()
            val_loss = 0
            val_f1 = 0
            val_acc = 0
            val_recall = 0
            val_count = 0
            val_precision = 0
            p_list = []
            t_list = []
            with torch.no_grad():
                for i, (input_ids, attention_mask, labels) in enumerate(val_loader):

                    out = model(input_ids=input_ids, attention_mask=attention_mask)

                    # out_z, labels_z = reshape_and_remove_pad(out, labels, attention_mask)

                    loss = -model.module.crf(out, labels, attention_mask.to(torch.bool))
                    loss = loss.sum() / val_loader.batch_size

                    out = model.module.crf.viterbi_decode(out, attention_mask.to(torch.bool))
                    # out = torch.argmax(out, dim=2)
                    # out = torch.where(out > threshold, torch.ones_like(out), torch.zeros_like(out))

                    true_labels = []

                    list5 = []
                    list6 = []
                    predicted_labels = out
                    for j in range(len(labels)):
                        true_label = labels[j].tolist()
                        t_first_index = true_label.index(207)
                        t_second_index = true_label.index(207, t_first_index + 1)
                        t_modified_label = true_label[t_first_index:t_second_index + 1]
                        true_labels.append(t_modified_label)

                    for z, j in zip(predicted_labels, true_labels):
                        list3 = []
                        list4 = []
                        for m, n in zip(z, j):
                            if m != 206 or n != 206:
                                list3.append(m)
                                list4.append(n)
                        list5.append(list3)
                        list6.append(list4)

                    y_true = [label for sublist in list6 for label in sublist]
                    y_pred = [label for sublist in list5 for label in sublist]
                    # accuracy = accuracy_score(true_labels, predicted_labels)
                    precision = precision_score(y_true, y_pred, average='micro')
                    recall = recall_score(y_true, y_pred, average='micro')
                    f1 = f1_score(y_true, y_pred, average='micro')

                    val_loss += loss.item()
                    val_f1 += f1
                    val_precision += precision
                    val_recall += recall
                    val_count += 1
                    # print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                    print(
                        f"第{epoch + 1}周期：第{i + 1}轮验证, loss：{loss.item()}, F1:{f1},precision:{precision},recall:{recall}")

                val_loss /= val_count
                val_f1 /= val_count
                val_precision /= val_count
                val_recall /= val_count

                print(
                    f"----------总第{epoch + 1}周期,loss为{val_loss},F1为{val_f1},precision为{val_precision}，recall为{val_recall}------------")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
                torch.save(model.state_dict(), "best_13.pt")
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping!")
                # torch.save(model.state_dict(), "early_3.pt")
                break
            lr_scheduler.step()
            print(f"学习率为{lr_scheduler.get_last_lr()}")
            print(f"counter为{counter}")
        print(f"验证集 F1 分数：{val_f1}")
        return best_val_f1, best_model_state

    except KeyboardInterrupt:
        print('手动终止训练')
        model_save_path = "c_model_interrupted_2.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"当前模型已保存到：{model_save_path}")


learning_rate = 1e-5
num_epochs = 2

test_f1, model_x = train_model(learning_rate, num_epochs)

model_save_path = "13.pth"
torch.save(model_x, model_save_path)
print(f"使用指定的超参数训练的模型已保存到：{model_save_path}")
print(test_f1)
