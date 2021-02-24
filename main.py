
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, classification_report
import random
from tqdm import tqdm
import pandas as pd
import json
import tensorflow as tf 
import keras.backend.tensorflow_backend as KTF

# 按需分配
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)


maxlen = 1536
epochs = 10
batch_size = 6

# bert配置
config_path = '/root/path/bert_config.json'
checkpoint_path = '/root/path/bert_model.ckpt'
dict_path = '/root/path/vocab.txt'


def load_data(path):
    data = json.loads(open(path).read())
    ret = []
    for item in data:
        content = item['Content']
        questions = item['Questions']
        for q in questions:
            question = q['Question']
            answer = q['Answer']
            qid = q['Q_id']
            choices = q['Choices']
            tmp = []
            for i in choices:
                if answer==i[0]:
                    label = 1
                    ret.append(
                        {
                            'content':content,
                            'question':question,
                            'qid':qid,
                            'choice':i[2:],
                            'label':label
                        }
                    )
                else:
                    label = 0
                    tmp.append(
                        {
                            'content':content,
                            'question':question,
                            'qid':qid,
                            'choice':i[2:],
                            'label':label
                        }
                    )
            ret.append(
                tmp[random.randint(0, len(tmp)-1)]
            )
    return ret
            

ret_data = load_data('/root/path/train.json')
random.shuffle(ret_data)
train_data = ret_data[:-5000]
valid_data = ret_data[-5000:]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            content = item['content']
            question = item['question']
            choice = item['choice']
            label = item['label']
            token_ids, segment_ids = tokenizer.encode(content+question, choice,maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    hierarchical_position=True, # 支持超长文本
    return_keras_model=False,
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(
    units=2, activation='softmax', kernel_initializer=bert.initializer
)(output)


model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    a,b = [],[]
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        a.extend([_ for _ in y_pred])
        b.extend([_ for _ in y_true])
        total += len(y_true)
        right += (y_true == y_pred).sum()
    report = classification_report(a,b)
    print(report)
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            json_config = model.to_json()
            with open('model_config_classification.json', 'w') as json_file:
                json_file.write(json_config)

            model.save_weights('model_weights_classification.h5')

        print(
            u'val_acc: %.5f, best_val_acc: %.5f' %
            (val_acc, self.best_val_acc)
        )



def main():
    evaluator = Evaluator()
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

def predict():
    with open('model_config_classification.json') as json_file:
        json_config = json_file.read()
    model = keras.models.model_from_json(json_config)
    model.load_weights('model_weights_classification.h5')
    
    valid_data = json.loads(open('/root/path/validation.json').read())
    Q_ID = []
    Q_Results = []
    id2label = {
        0:"A",
        1:"B",
        2:"C",
        3:"D"
    }
    for item in tqdm(valid_data):
        content = item['Content']
        questions = item['Questions']
        for q in questions:
            question = q['Question']
            qid = q['Q_id']
            choices = q['Choices']
            tmp = []
            for c in choices:
                c = c[2:]
                token_ids, segment_ids = tokenizer.encode(content+question, c, maxlen=maxlen)
                y_pred = model.predict([[token_ids], [segment_ids]])
                tmp.append(y_pred.max(1)[0])
            Q_ID.append(qid)
            Q_Results.append(id2label[np.array(tmp).argmax()])
    df = pd.DataFrame({'id':Q_ID,'label':Q_Results})
    df.to_csv('./submit.csv',index=None)          
            
                    
if __name__=='__main__':
    import fire
    fire.Fire()