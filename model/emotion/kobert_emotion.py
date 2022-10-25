import torch
import numpy as np
import gluonnlp as nlp
from kobert.utils import get_tokenizer

from model.emotion import classifier
from model.emotion import dataloader


# 모델
bertmodel, vocab = get_pytorch_kobert_model()
model = classifier.EmotionClassifier(bertmodel, dr_rate=0.5)

# 토크나이저
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# gpu/cpu
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# 모델 불러오기
checkpoint = torch.load('../../checkpoint/kobert.pth', map_location=device)
model.load_state_dict(checkpoint, strict=False)

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


def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_new = [data]

    max_len = dataloader.max_len
    batch_size = dataloader.batch_size

    new_test = dataloader.EmotionDataset(dataset_new, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(new_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            return flip_emotion_labels[np.argmax(logits)]
