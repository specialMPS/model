import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

from preprocess.train_gpt import KoGPT2Chat

import kss
from eunjeon import Mecab
from hanspell import spell_checker

mecab = Mecab()
# from konlpy.tag import Kkma
#
# kkma = Kkma()

parser = argparse.ArgumentParser(description='Simsini based on KoGPT-2')

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
                    default='././checkpoint/model_-last.ckpt',
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
        # 1. kss로 문장분류
        s = input_sentence
        bf_sent = ''  # 이전 문장
        sent_len = len(kss.split_sentences(s))  # 쪼개진 문장 전체 길이
        for i, sent in enumerate(kss.split_sentences(s), start=1):  # 쪼개진 문장 중에 마지막 문장만 가져오기
            if sent_len != i:
                bf_sent = sent
            else:
                mpos = mecab.pos(input_sentence)
                for index, value in enumerate(mpos):
                    if 'SY' in value[1] or 'SC' in value[1]:
                        input_sentence = bf_sent
                    elif 'MAG' == value[1]:
                        input_sentence = bf_sent
                    else:
                        input_sentence = sent
        print("kss : {}".format(input_sentence))

        # 2. hanspell로 문장 맞춤법 검사
        result = spell_checker.check(input_sentence)
        result = result.as_dict()  # dict 형태로 변환
        input_sentence = result['checked']
        print("hanspell : {}".format(input_sentence))

        # 3. mecab을 이용해 형태소 분석 후 되묻는 질문 만들기
        # total check
        jjx = False

        m_pos = mecab.pos(input_sentence)
        print(input_sentence)
        print(m_pos)

        for index, value in enumerate(m_pos):
            # 처음 인사할 때 제외
            if '안녕' in value[0] or '안녕하세요' in value[0] or '하이' in value[0]:
                jjx = True
                break
            elif '고마워' in value[0] or '고마워요' in value[0]:
                jjx = True
                break
            # 목적어가 생략되어 있는지 확인, 목적어가 없다고 하더라도 다른 형태가 목적어를 대신하고 있는지 확인
            # 목적격 조사
            elif 'JKO' in value[1]:
                jjx = True
                print('jko true')
                break
            # 주격 조사
            elif 'JKS' in value[1]:
                jjx = True
                print('jks true')
                break
            # 보격 조사
            elif 'JKC' in value[1]:
                jjx = True
                print('jkc true')
                break
            # 형용사 파생 접미사
            elif 'XSA' in value[1]:
                jjx = True
                print('xsa true')
                break
            # 마침표, 물음표, 느낌표
            elif 'SF' in value[1] and '?' in value[0]:
                jjx = True
                break
            elif '모르' in value[0] or '몰라' in value[0]:
                jjx = True
            # 접속 부사
            elif 'MAJ' in value[1]:
                jjx = True
                break
            else:
                print('jjx false')

        # 목적어가 생략된 문장에서 다시 되물어 보기 위해 질문 만들기
        # for문 제일 마지막에 왔을 때
        ind = len(m_pos)-1
        if jjx == False:
            # 마지막에 마침표 물음표 느낌표가 있는지 확인
            # 있다면 index를 1을 줄여서 확인
            # if 'SF' in m_pos[ind][1]:
            #     print('sf')
            #     ind = ind - 1
            # if 'EC' in m_pos[ind][1] or 'EF' in m_pos[ind][1]:
            #     ind = ind - 1
            # if 'EP' in m_pos[ind][1]:
            #     print('ep')
            #     if 'VV' in m_pos[ind - 1] or 'VA' in m_pos[ind - 1] or 'XR' in m_pos[ind - 1]:
            #         answer = "왜 " + m_pos[ind - 1][0] + m_pos[ind][0] + "어요??"
            #         return answer
            #     else:
            #         answer = "왜 " + m_pos[ind][0] + "어요??"
            #         return answer
            answer = "좀 더 자세히 말해줄 수 있나요?"
            return answer
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
