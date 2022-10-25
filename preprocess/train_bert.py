# import gc
# import os
#
# import numpy as np
# import torch
# from torch.utils.data import dataloader
# from tqdm import tqdm
# from transformers import AdamW
#
# from model.emotion.classifier import KoBERTforEmotionClassification
# from model.emotion.dataloader import EmotionDataset
#
#
# def train(device, epoch, model, optimizer, train_loader, save_step, save_ckpt_path, train_step=0):
#     losses = []
#     train_start_index = train_step + 1 if train_step != 0 else 0
#     total_train_step = len(train_loader)
#     model.train()
#
#     with tqdm(total=total_train_step, desc=f"Train({epoch})") as pbar:
#         pbar.update(train_step)
#         for i, data in enumerate(train_loader, train_start_index):
#
#             optimizer.zero_grad()
#             outputs = model(**data)
#
#             loss = outputs[0]
#
#             losses.append(loss.item())
#
#             loss.backward()
#             optimizer.step()
#
#             pbar.update(1)
#             pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")
#
#             if i >= total_train_step or i % save_step == 0:
#                 torch.save({
#                     'epoch': epoch,  # 현재 학습 epoch
#                     'model_state_dict': model.state_dict(),  # 모델 저장
#                     'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
#                     'loss': loss.item(),  # Loss 저장
#                     'train_step': i,  # 현재 진행한 학습
#                     'total_train_step': len(train_loader)  # 현재 epoch에 학습할 총 train step
#                 }, save_ckpt_path)
#
#     return np.mean(losses)
#
#
# if __name__ == '__main__':
#     gc.collect()
#     torch.cuda.empty_cache()
#
#     data_path = "../data/data_final_cleaned.csv"
#     checkpoint_path = "../checkpoint"
#     save_ckpt_path = f"{checkpoint_path}/kobert.pth"
#
#     n_epoch = 20
#     batch_size = 64
#     ctx = "cuda" if torch.cuda.is_available() else "cpu"
#     device = torch.device(ctx)
#     save_step = 100  # 학습 저장 주기
#     learning_rate = 5e-5  # Learning Rate
#
#     # EmotionDataset Data Loader
#     dataset = EmotionDataset(file_path=data_path, device=device)
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     model = KoBERTforEmotionClassification()
#     model.to(device)
#
#     # Prepare optimizer and schedule (linear warmup and decay)
#     no_decay = ['bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#          'weight_decay': 0.01},
#         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
#
#     pre_epoch, pre_loss, train_step = 0, 0, 0
#     if os.path.isfile(save_ckpt_path):
#         checkpoint = torch.load(save_ckpt_path, map_location=device)
#         pre_epoch = checkpoint['epoch']
#         train_step = checkpoint['train_step']
#         total_train_step = checkpoint['total_train_step']
#
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#         print(f"load pretrain from: {save_ckpt_path}, epoch={pre_epoch}")
#
#     losses = []
#     offset = pre_epoch
#     for step in range(n_epoch):
#         epoch = step + offset
#         loss = train(device, epoch, model, optimizer, train_loader, save_step, save_ckpt_path, train_step)
#         losses.append(loss)
