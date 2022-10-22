import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

from preprocess.train_gpt import KoGPT2Chat


# from eunjeon import Mecab

# mecab = Mecab()
from konlpy.tag import Kkma

kkma = Kkma()

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params',
                    type=str,
                    default='././checkpoint/model_-last_test.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')


U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                                                    pad_token=PAD, mask_token=MASK)


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        # self.hparams = hparams # read file
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def chat(self, input_sentence, sent='0'):
        # test mecab
        # mecab으로 품사 태깅
        # test kkma
        # kkma로 품사 태깅
        # total check
        jjx = False

        k_pos = kkma.pos(input_sentence)
        # m_pos = mecab.pos(input_sentence)
        print(input_sentence)

        for index, value in enumerate(k_pos):
            # 처음 인사할 때 제외
            if input_sentence == '안녕' or input_sentence == '안녕하세요':
                jjx = True
                break
            # 목적어가 생략되어 있는지 확인, 목적어가 없다고 하더라도 다른 형태가 목적어를 대신하고 있는지 확인
            elif 'JKO' in value[1]:
                jjx = True
                print('jko true')
                break
            elif 'JKS' in value[1]:
                jjx = True
                print('jks true')
                break
            elif 'JKC' in value[1]:
                jjx = True
                print('jkc true')
                break
            elif 'XSA' in value[1]:
                jjx = True
                print('xsa true')
                break
            elif 'SF' in value[1] and '?' in value[0]:
                jjx = True
            # elif '모르' in value[0]:
            #     jjx = True
            else:
                print('jjx false')

        # 목적어가 생략된 문장에서 다시 되물어 보기 위해 질문 만들기
        # for문 제일 마지막에 왔을 때
        ind = len(k_pos)-1
        if jjx == False:
            # 마지막에 마침표 물음표 느낌표가 있는지 확인
            # 있다면 index를 1을 줄여서 확인
            if 'SF' in k_pos[ind][1]:
                print('sf')
                ind = ind - 1
            if 'EC' in k_pos[ind][1] or 'EF' in k_pos[ind][1]:
                ind = ind - 1
            if 'EP' in k_pos[ind][1]:
                print('ep')
                if 'VV' in k_pos[ind - 1] or 'VA' in k_pos[ind - 1] or 'XR' in k_pos[ind - 1]:
                    answer = "왜 " + k_pos[ind - 1][0] + k_pos[ind][0] + "어요??"
                    return answer
                else:
                    answer = "왜 " + k_pos[ind][0] + "어요??"
                    return answer
            # answer = "왜요?"
            # return answer
            jjx = True

        if jjx == True:
            tok = TOKENIZER
            sent_tokens = tok.tokenize(sent)
            print(sent_tokens)
            with torch.no_grad():
                q = input_sentence.strip()
                a = ''
                while 1:
                    input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                    pred = self(input_ids)
                    gen = tok.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                    # print(gen) # <pad>
                    if gen == EOS or gen == PAD: # PAD 무한 루프 에러 방지
                        break
                    a += gen.replace('▁', ' ')

                a = a.strip()
                # period_pos = a.rfind(".")
                # question_pos = a.rfind("?")
                # exclamation_pos = a.rfind("!")
                # last_pos = len(a) - 1
                # if last_pos == period_pos or last_pos == question_pos or last_pos == exclamation_pos:
                #     return a
                # mark_pos = max(max(period_pos, question_pos), exclamation_pos)
                # a = a[:mark_pos + 1]
                if a == "":
                    return "듣고 있어요. 계속 얘기해주세요!"
                return a

parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

model = KoGPT2Chat(args)
model = model.load_from_checkpoint(args.model_params)

def predict(sent):
    return model.chat(sent)
