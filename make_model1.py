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
import torchaudio.transforms as T
torch.backends.cuda.enable_flash_sdp(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


root = tk.Tk()
root.title("5_1")
img_label4 = tk.Label(root, width=1300, height=100, bg="black")
img_label5 = tk.Label(root, width=1300, height=100, bg="black")
img_label6 = tk.Label(root, width=1300, height=100, bg="black")
img_label7 = tk.Label(root, width=1300, height=100, bg="black")


img_label4.grid(row=0, column=0)
img_label5.grid(row=1, column=0)
img_label6.grid(row=2, column=0)
img_label7.grid(row=3, column=0)


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
class Musiclm1(nn.Module):
    def __init__(self):
        super().__init__()
        self.charset = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄶㄳㄵㄺㄼㄽㄾㄿㅀㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣabcdefghijklmnopqrstuvwxyz1234567890`'\"\\?/><.,!()~@#$%^&*-_+= \n\r"    
        n_symbols = len(self.charset) + 2
        d_model =512
        n_mel_channels = 80
        self.n_mel_channels = n_mel_channels
        self.positinoal_encoding = PositionalEncoding(d_model, max_len=250*8)  
        #8000 4000 2000 1000 500

        self.transformer4s_1 = nn.ModuleList([nn.Transformer(d_model, 8, 1, 1, batch_first=True) for _ in range(12)])
      
        self.embedding3 = nn.Embedding(n_symbols, 80)#160)
        self.text_prenet3 = myPrenet()
        self.prenet3 =  myPrenet()
      
        self.linear_projection3 = nn.Sequential(
            nn.ConvTranspose1d(512, 320, 4, 2, 1),  
            nn.LeakyReLU(),
            nn.ConvTranspose1d(320, 160, 4, 2, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(160, 80, 4, 2, 1) ,
            nn.Tanh()
        )
                        
  
    def forward(self, x, y2):
        x2 = self.embedding3(x)
        y= self.prenet3(y2)
        x2 = self.text_prenet3(x2)

        out = x2 
        for transformer in self.transformer4s_1:
            out1 = transformer(self.positinoal_encoding(y), self.positinoal_encoding(out))
            out = out1 + out
        mel3 = self.linear_projection3(out.permute(0,2,1)).permute(0,2,1)
        return mel3
    
criterion = MelSpectrogramDetailLoss()


g_model1 = Musiclm1().to(device)
g_model1 = torch.load(
   "./pth_save/1g475000.pt",
   weights_only=False,
)
optimizerG = torch.optim.Adam(g_model1.parameters(), lr=1e-4)#, betas=(0.0, 0.99))

while_number = 475000
encoding_texts_batch2 =np.load(f"./np_data/encoding_texts_batch.npy")
break_batch2 = np.load(f"./np_data/break_batch.npy")
accompaniment_batch2 = np.load(f"./np_data/accompaniment_batch.npy")
song_batch2 = np.load(f"./np_data/song_batch.npy")
encoding_texts_batch =np.load(f"./np_data_1/encoding_texts_batch.npy")
break_batch = np.load(f"./np_data_1/break_batch.npy")
accompaniment_batch = np.load(f"./np_data_1/accompaniment_batch.npy")
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
                
        
                fake_data4 = g_model1.forward(string_data, accompaniment_music_data) 


                l1 = criterion(fake_data4, song_data)
                mr_mel = multi_resolution_mel_loss(fake_data4, song_data)
                
                
                loss_G = loss_g = (
                    l1
                    + mr_mel
                )
                optimizerG.zero_grad()
                loss_G.backward()            
                optimizerG.step()

                print(f"step_number : {while_number}, loss_value :    {loss_G.item()}   ")
                if while_number % 101 == 1:
                    g_model1.eval()
               
                    with torch.no_grad():       
                        test_num = 0
                    
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            g_fake_data4 = g_model1(string_data[0:1], accompaniment_music_data[0:1])
                                        
                           
                            g_fake_datas4 = (g_fake_data4[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            fake_datas4 = (fake_data4[0].permute(1,0)  * 255).float().cpu().numpy().astype(np.uint8)
                            accompaniment_music_datas = (accompaniment_music_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            song_datas = (song_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                        
                            img = Image.fromarray(accompaniment_music_datas)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label4.config(image=photo)
                            img_label4.image = photo 

                            img = Image.fromarray(g_fake_datas4)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label5.config(image=photo)
                            img_label5.image = photo  # 가비지 컬렉션 방
                            
                            img = Image.fromarray(fake_datas4)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label6.config(image=photo)
                            img_label6.image = photo  # 가비지 컬렉션 방지

                            img = Image.fromarray(song_datas)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label7.config(image=photo)
                            img_label7.image = photo 


            
                            root.update()
                
                if while_number % 5000 == 0:
                    torch.save(g_model1, f"./pth_save/1g{while_number}.pt")
                   
                    

                
