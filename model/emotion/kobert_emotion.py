import torch
import numpy as np
import gluonnlp as nlp
from torch.utils.data import Dataset

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from model.emotion import classifier


class EmotionDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i] + (self.labels[i], )

    def __len__(self):
        return len(self.labels)


# flipped emotion dict
emotion_labels = {'중립': 0,
                  '기쁨': 1,
                  '놀람': 2,
                  '긴장됨': 3,
                  '괴로움': 4,
                  '화남': 5,
                  '비참함': 6,
                  '우울함': 7,
                  '피로함': 8}
flip_emotion_labels = {v: k for k, v in emotion_labels.items()}

# 모델
bertmodel, vocab = get_pytorch_kobert_model()
model = classifier.EmotionClassifier(bertmodel, dr_rate=0.5)

# 토크나이저
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# gpu/cpu
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# 모델 불러오기 - 방법 1
model = torch.load('../../checkpoint/kobert_v2.pt', map_location=device)
model.load_state_dict(torch.load('../../checkpoint/kobert_v2_state_dict.pt'))

# 모델 불러오기 - 방법 2
checkpoint = torch.load('../../checkpoint/kobert_v2.pth', map_location=device)
model.load_state_dict(checkpoint, strict=False)

model.eval()


def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_new = [data]

    max_length = 64
    b_size = 32

    new_test = EmotionDataset(dataset_new, 0, 1, tok, max_length, True, False)
    test_dataloader = torch.utils.data.DataLoader(new_test, batch_size=b_size, num_workers=5)

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long()
        segment_ids = segment_ids.long()

        valid_length = valid_length
        label = label.long()

        out = model(token_ids, valid_length, segment_ids)

        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            logits_zero = [np.round(x, 3) if x > 0 else 0 for x in logits]
            print(logits_zero)

            return flip_emotion_labels[np.argmax(logits)]
