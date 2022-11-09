# import numpy as np
# import pandas as pd
# import gluonnlp as nlp
# import torch
# from torch.utils.data import Dataset
# from kobert.utils import get_tokenizer
# from kobert.pytorch_kobert import get_pytorch_kobert_model
# from sklearn.model_selection import train_test_split
#
#
# class EmotionDataset(Dataset):
#     def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
#         transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)
#
#         self.sentences = [transform([i[sent_idx]]) for i in dataset]
#         self.labels = [np.int32(i[label_idx]) for i in dataset]
#
#     def __getitem__(self, i):
#         return self.sentences[i] + (self.labels[i], )
#
#     def __len__(self):
#         return len(self.labels)
#
#
# if __name__ == '__main__':
#     dataset = pd.read_csv('../../data/data_final_cleaned.csv')
#
#     emotion_labels = {'중립': 0,
#                       '기쁨': 1,
#                       '놀람': 2,
#                       '긴장됨': 3,
#                       '괴로움': 4,
#                       '화남': 5,
#                       '비참함': 6,
#                       '우울함': 7,
#                       '피로함': 8}
#
#     data_list = []
#     for q, label in zip(dataset['sentence'], dataset['emotion']):
#         data = [q, str(emotion_labels[label])]
#         data_list.append(data)
#
#     # train/test 분리
#     train, test = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=42)
#
#     # parameters
#     max_len = 256
#     batch_size = 64
#     warmup_ratio = 0.1
#     num_epochs = 10
#     max_grad_norm = 1
#     log_interval = 200
#     learning_rate = 5e-5
#
#     # model, tokenizer
#     bertmodel, vocab = get_pytorch_kobert_model()
#     tokenizer = get_tokenizer()
#     tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
#
#     # tokenizing
#     data_train = EmotionDataset(train, 0, 1, tok, vocab, max_len, True, False)
#     data_test = EmotionDataset(test, 0, 1, tok, vocab, max_len, True, False)
#
#     # make data to torch type
#     train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
#     test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)
