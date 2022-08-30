import torch
import torch.nn as nn
import random

from model.model.emotion.classifier import KoBERTforSequenceClassification
from kobert_transformers import get_tokenizer


def load_wellness_answer():
    category_path = "././data/emotion_category.txt"
    c_f = open(category_path, 'r')
    category_lines = c_f.readlines()
    category = {}

    for line_num, line_data in enumerate(category_lines):
        data = line_data.split('    ')
        category[data[1][:-1]] = data[0]

    return category


def kobert_input(tokenizer, str, device=None, max_seq_len=512):
    index_of_words = tokenizer.encode(str)
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
    checkpoint_path = "././checkpoint"
    save_ckpt_path = f"{checkpoint_path}/kobert_wellness.pth"

    # 답변과 카테고리 불러오기
    category = load_wellness_answer()

    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)

    # 저장한 Checkpoint 불러오기
    checkpoint = torch.load(save_ckpt_path, map_location=device)

    model = KoBERTforSequenceClassification()
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(ctx)
    model.eval()

    tokenizer = get_tokenizer()

    while 1:
        sent = input('\nQuestion: ')  # '요즘 기분이 우울한 느낌이에요'
        data = kobert_input(tokenizer, sent, device, 512)

        if '종료' in sent:
            break

        output = model(**data)

        logit = output[0]
        softmax_logit = torch.softmax(logit, dim=-1)
        softmax_logit = softmax_logit.squeeze()

        max_index = torch.argmax(softmax_logit).item()
        max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

        emotion = category[str(max_index)]

        print(f'index: {max_index}, emotion: {emotion}, softmax_value: {max_index_value}')
        print('-' * 50)