import torch
import torch.nn as nn
# from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
# import math
import torch.nn.functional as F
# from itertools import chain
from modules import Postnet, TextPrenet, ConvNorm, Postnet, PositionalEncoding, Prenet

torch.backends.cuda.enable_flash_sdp(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_MELS = 80
LAMBDA_L1 = 40.0
LAMBDA_MSE = 2.0
LAMBDA_FM = 10.0
LAMBDA_ADV = 1.0

root = tk.Tk()
root.title("5_0copy")
img_label1 = tk.Label(root, width=1300, height=100, bg="black")
# img_label2 = tk.Label(root, width=1300, height=50, bg="black")
img_label3 = tk.Label(root, width=1300, height=100, bg="black")
img_label4 = tk.Label(root, width=1300, height=100, bg="black")
img_label5 = tk.Label(root, width=1300, height=100, bg="black")


img_label1.grid(row=0, column=0)
# img_label2.grid(row=1, column=0)
img_label3.grid(row=1, column=0)
img_label4.grid(row=2, column=0)
img_label5.grid(row=3, column=0)


class MelSpectrogramDetailLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        """
        pred, target shape: [Batch, Time_Length(L), Mel_Bins(E)]
        """
        # 1) 전체적인 데시벨 에너지 분포를 맞추는 베이스라인 Loss
        l1 = self.l1_loss(pred, target)
        
        # 2) 강한 피크나 급격한 음압 변화를 세게 잡아주는 L2 Loss
        l2 = self.mse_loss(pred, target)
        
        # ----------------------------------------------------
        # 미세 잡음(Stochastic Noise) 및 질감 복원을 위한 차분(Gradient) 손실
        # ----------------------------------------------------
        # 3) 시간 축(1번 축, L) 기준 차분: 시간 흐름에 따른 미세 진동, 노이즈의 변화 감지
        diff_time_pred = pred[:, 1:, :] - pred[:, :-1, :]
        diff_time_target = target[:, 1:, :] - target[:, :-1, :]
        time_grad_loss = self.l1_loss(diff_time_pred, diff_time_target)
        
        # 4) 주파수 축(2번 축, E) 기준 차분: 자글자글한 고주파수 잡음의 형태 강제 학습
        diff_freq_pred = pred[:, :, 1:] - pred[:, :, :-1]
        diff_freq_target = target[:, :, 1:] - target[:, :, :-1]
        freq_grad_loss = self.l1_loss(diff_freq_pred, diff_freq_target)

        # 최종 가중치 합산 (디테일과 잡음 성분을 위해 차분 손실에 비중을 둡니다)
        total_loss = l1 + 0.2 * l2 + 0.5 * time_grad_loss + 0.5 * freq_grad_loss
        return total_loss


def multi_resolution_mel_loss(fake, real):
    """
    Simple multi-resolution mel loss: compare at full, half, and quarter time
    resolutions. This is NOT a waveform STFT loss; it is appropriate as an
    additional mel-domain consistency term.
    """

    fake = fake.permute(0,2,1)
    real = real.permute(0,2,1)
    total = 0.0
    for factor in (1, 2, 4):
        if factor == 1:
            f, r = fake, real
        else:
            f = F.avg_pool1d(fake, kernel_size=factor, stride=factor,
                             ceil_mode=False)
            r = F.avg_pool1d(real, kernel_size=factor, stride=factor,
                             ceil_mode=False)
        total = total + F.l1_loss(f, r)
    return total / 3.0

class myPrenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(80, 160, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv1d(160, 320, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv1d(320, 512, 4, 2, 1),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0,2,1)
        return x

# class Noise(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(
#                     nn.Conv1d(80, 160, 4, 2, 1),
#                     nn.LeakyReLU(),
#                     nn.Conv1d(160, 320, 4, 2, 1),
#                     nn.LeakyReLU(),
#                     nn.Conv1d(320, 512, 4, 2, 1),
#                     nn.LeakyReLU(),
#                 )
#         self.linear_projection1 = nn.Sequential(
#                     nn.ConvTranspose1d(512, 320, 4, 2, 1),  
#                     nn.LeakyReLU(),
#                     nn.ConvTranspose1d(320, 160, 4, 2, 1),
#                     nn.LeakyReLU(),
#                     nn.ConvTranspose1d(160, 80, 4, 2, 1) ,
#                 )
#     def forward(self, x):
#         x = x.permute(0,2,1)
#         x = self.linear_projection1(self.conv(x)).permute(0,2,1)
#         return x
class Musiclm0(nn.Module):
    def __init__(self):
        super().__init__()
        self.charset = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄶㄳㄵㄺㄼㄽㄾㄿㅀㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣabcdefghijklmnopqrstuvwxyz1234567890`'\"\\?/><.,!()~@#$%^&*-_+= \n\r"    
        n_symbols = len(self.charset) + 2
        d_model =512
        n_mel_channels = 80
        self.n_mel_channels = n_mel_channels
        self.positinoal_encoding = PositionalEncoding(d_model, max_len=250*4)  
        #8000 4000 2000 1000 500

        # self.transformer1s_1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, batch_first=True), 1)
        # self.transformer1s_2 = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.transformer1s_3 = nn.ModuleList([nn.Transformer(d_model, 8, 1, 1, batch_first=True) for _ in range(12)])
        # self.transformer1s_4 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, batch_first=True), 1)
        self.embedding1 = nn.Embedding(n_symbols, 80)#160)
        self.text_prenet1 = myPrenet()
        self.prenet1 =  myPrenet()
      
        self.linear_projection1 = nn.Sequential(
            nn.ConvTranspose1d(512, 320, 4, 2, 1),  
            nn.LeakyReLU(),
            nn.ConvTranspose1d(320, 160, 4, 2, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(160, 80, 4, 2, 1) ,
            nn.Tanh()
        )
                        
       
    def forward(self, x, y2):
        x2 = self.embedding1(x)
        y= self.prenet1(y2)
        x2 = self.text_prenet1(x2)

        out=y
        for transformer in self.transformer1s_3:
            out1 = transformer(self.positinoal_encoding(x2), self.positinoal_encoding(out))
            out = out1 + out
        mel1 = self.linear_projection1(out.permute(0,2,1)).permute(0,2,1)
        return mel1
criterion = MelSpectrogramDetailLoss()
g_model1 = Musiclm0().to(device)
# g_model2 = Noise().to(device)
g_model1 = torch.load(
   "./pth_save/0g465000.pt",
   weights_only=False,
)

# g_model2 = torch.load(
#    "./pth_save/02g160000.pt",
#    weights_only=False,
# )

# d_model2 = torch.load(
#    "./pth_save/1d2120000.pt",
#    weights_only=False,
# )


optimizerG = torch.optim.Adam(g_model1.parameters(), lr=1e-4)
# optimizerNG = torch.optim.Adam(g_model2.parameters(), lr=1e-5)

while_number = 465000
encoding_texts_batch2 =np.load(f"./np_data/encoding_texts_batch.npy")
break_batch2 = np.load(f"./np_data/break_batch.npy")
accompaniment_batch2 = np.load(f"./np_data/accompaniment_batch.npy")
# origin_batch = np.load(f"./np_data/origin_batch.npy")
song_batch2 = np.load(f"./np_data/song_batch.npy")
encoding_texts_batch =np.load(f"./np_data_1/encoding_texts_batch.npy")
break_batch = np.load(f"./np_data_1/break_batch.npy")
accompaniment_batch = np.load(f"./np_data_1/accompaniment_batch.npy")
# origin_batch = np.load(f"./np_data/origin_batch.npy")
song_batch = np.load(f"./np_data_1/song_batch.npy")

encoding_texts_batch = np.concatenate([encoding_texts_batch, encoding_texts_batch2], axis=0)
break_batch = np.concatenate([break_batch, break_batch2], axis=0)
accompaniment_batch = np.concatenate([accompaniment_batch, accompaniment_batch2], axis=0)
song_batch = np.concatenate([song_batch, song_batch2], axis=0)

while True:
    
        for epoch in range(0, song_batch.shape[0], 5):
            
                g_model1.train()
                # g_model2.train()
             
                while_number += 1
                string_data=torch.from_numpy(encoding_texts_batch[epoch:epoch+5]).type(torch.long).to(device)
                breaking_music_data=torch.from_numpy(break_batch[epoch:epoch+5]).type(torch.float).to(device)
                accompaniment_music_data=torch.from_numpy(accompaniment_batch[epoch:epoch+5]).type(torch.float).to(device)
                song_data=torch.from_numpy(song_batch[epoch:epoch+5]).type(torch.float).to(device)

     
                fake_data4 = g_model1.forward(string_data, breaking_music_data) 

                l1 = criterion(fake_data4, accompaniment_music_data)
                mr_mel = multi_resolution_mel_loss(fake_data4, accompaniment_music_data)
                
                
                loss_G = loss_g = (
                    l1
                    + mr_mel
                )

                optimizerG.zero_grad()
                loss_G.backward()
                optimizerG.step()

                # fake_data4 = g_model1(string_data, breaking_music_data) 
                # noize_out = g_model2(fake_data4.detach())
                # noize = accompaniment_music_data - fake_data4.detach()
                # noize_mse = F.mse_loss(noize_out, noize)
               
                # optimizerNG.zero_grad()
                # noize_mse.backward()
                # optimizerNG.step()
                
                print(f"step_number : {while_number}, loss_value :    {loss_G.item()}")#  {noize_mse.item()} ")
                if while_number % 101 == 1:
                    g_model1.eval()
                    # g_model2.eval()
                  
                    with torch.no_grad():       
                        test_num = 0
                        # with sdpa_kernel(
                        #     [SDPBackend.FLASH_ATTENTION]
                        # ):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            g_fake_data4 = g_model1(string_data[0:1], breaking_music_data[0:1])
                            # g_fake_data4_for_noize = g_fake_data4
                            # for _ in range(5):
                            #     noize = g_model2(g_fake_data4_for_noize)
                            #     g_fake_data4_for_noize = g_fake_data4_for_noize + noize
                            
                            g_fake_datas4 = (g_fake_data4[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            
                            fake_datas4 = (fake_data4[0].permute(1,0)  * 255).float().cpu().numpy().astype(np.uint8)
                            breaking_music_datas = (breaking_music_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            accompaniment_music_datas = (accompaniment_music_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            # g_fake_data4_for_noize_datas  = (g_fake_data4_for_noize[0].permute(1, 0)*255).float().cpu().numpy().astype(np.uint8)
                            
                            img = Image.fromarray(breaking_music_datas)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label1.config(image=photo)
                            img_label1.image = photo 


                            # img = Image.fromarray(g_fake_data4_for_noize_datas)
                            # img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            # img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            # photo = ImageTk.PhotoImage(img)
                            # img_label2.config(image=photo)
                            # img_label2.image = photo  # 가비지 컬렉션 방
                            
                            img = Image.fromarray(g_fake_datas4)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label3.config(image=photo)
                            img_label3.image = photo  # 가비지 컬렉션 방
                            
                            img = Image.fromarray(fake_datas4)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label4.config(image=photo)
                            img_label4.image = photo  # 가비지 컬렉션 방지

                            img = Image.fromarray(accompaniment_music_datas)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label5.config(image=photo)
                            img_label5.image = photo 

                            
                            root.update()
                
                if while_number % 5000 == 0:
                    torch.save(g_model1, f"./pth_save/0g{while_number}.pt")
                    # torch.save(g_model2, f"./pth_save/02g{while_number}.pt")
                    

                
