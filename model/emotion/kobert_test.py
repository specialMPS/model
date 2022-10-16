import torch
import torch.nn as nn
import random

from model.emotion.classifier import KoBERTforEmotionClassification
from kobert_transformers import get_tokenizer


def kobert_input(tok, str, device=None, max_seq_len=128):
    index_of_words = tok.encode(str)
    token_type_ids = [0] * len(index_of_words)
    attention_mask = [1] * len(index_of_words)

    # Padding Length
    padding_length = max_seq_len - len(index_of_words)

    # Zero Padding
    index_of_words += [0] * padding_length
    token_type_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    data = {
        'input_ids': torch.tensor([index_of_words]).to(device),
        'token_type_ids': torch.tensor([token_type_ids]).to(device),
        'attention_mask': torch.tensor([attention_mask]).to(device),
    }

    return data


if __name__ == "__main__":

    checkpoint_path = "../../checkpoint"
    save_ckpt_path = f"{checkpoint_path}/kobert.pth"

    # 답변과 카테고리 불러 오기
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

    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)

    # 저장한 Checkpoint 불러 오기
    checkpoint = torch.load(save_ckpt_path, map_location=device)

    model = KoBERTforEmotionClassification()
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(ctx)
    model.eval()

    tokenizer = get_tokenizer()

    while 1:
        sent = input('\nSentence: ')  # '요즘 기분이 우울한 느낌이에요'
        data = kobert_input(tokenizer, sent, device, 128)

        if '종료' in sent:
            break

        output = model(**data)

        logit = output[0]
        softmax_logit = torch.softmax(logit, dim=-1)
        softmax_logit = softmax_logit.squeeze()

        max_index = torch.argmax(softmax_logit).item()
        max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

        emotion = flip_emotion_labels[max_index]

        print(f'index: {max_index}, emotion: {emotion}, softmax_value: {max_index_value}')
        print('-' * 50)
