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
import __main__  
torch.backends.cuda.enable_flash_sdp(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

root = tk.Tk()
root.title("5_0copy")
img_label1 = tk.Label(root, width=1300, height=100, bg="black")
img_label2 = tk.Label(root, width=1300, height=100, bg="black")
img_label3 = tk.Label(root, width=1300, height=100, bg="black")
img_label4 = tk.Label(root, width=1300, height=100, bg="black")
img_label5 = tk.Label(root, width=1300, height=100, bg="black")
img_label6 = tk.Label(root, width=1300, height=100, bg="black")
img_label7 = tk.Label(root, width=1300, height=100, bg="black")
img_label8 = tk.Label(root, width=1300, height=100, bg="black")
img_label9 = tk.Label(root, width=1300, height=100, bg="black")
img_label10 = tk.Label(root, width=1300, height=100, bg="black")

img_label1.grid(row=0, column=0)
img_label2.grid(row=1, column=0)
img_label3.grid(row=2, column=0)
img_label4.grid(row=3, column=0)
img_label5.grid(row=4, column=0)
img_label6.grid(row=5, column=0)
img_label7.grid(row=6, column=0)
img_label8.grid(row=7, column=0)
img_label9.grid(row=8, column=0)
img_label10.grid(row=9, column=0)


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
        self.transformer1s_3 = nn.ModuleList([nn.Transformer(d_model, 8, 1, 1,  batch_first=True) for _ in range(8)])
        self.transformer1s_2 = nn.ModuleList([nn.Transformer(d_model, 8, 1, 1,  batch_first=True) for _ in range(8)])
        # self.transformer1s_4 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, batch_first=True), 1)
        self.embedding1 = nn.Embedding(n_symbols, 80)#160)
        self.text_prenet1 = myPrenet()
        self.text_prenet2 = myPrenet()
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
        
        x_1 = self.text_prenet1(x2)
        x_2 = self.text_prenet2(x2)
                
        out_1 = self.positinoal_encoding(x_1)
        y_1 = self.positinoal_encoding(y)
        for transformer in self.transformer1s_2:
            out1 = transformer(y_1, out_1)
            out_1 = out1 + out_1

        out_1 = self.positinoal_encoding(out_1)
        x_2=self.positinoal_encoding(x_2)
        for transformer in self.transformer1s_3:
            out1 = transformer(x_2 ,out_1)
            out_1 = out1 + out_1
        mel1 = self.linear_projection1(out_1.permute(0,2,1)).permute(0,2,1)
        return mel1
class Musiclm1(nn.Module):
    def __init__(self):
        super().__init__()
        self.charset = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄶㄳㄵㄺㄼㄽㄾㄿㅀㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣabcdefghijklmnopqrstuvwxyz1234567890`'\"\\?/><.,!()~@#$%^&*-_+= \n\r"    
        n_symbols = len(self.charset) + 2
        d_model =512
        n_mel_channels = 80
        self.n_mel_channels = n_mel_channels
        self.positinoal_encoding = PositionalEncoding(d_model, max_len=250*4)  
        #8000 4000 2000 1000 500

        self.transformer1s_3 = nn.ModuleList([nn.Transformer(d_model, 8, 1, 1,  batch_first=True) for _ in range(8)])
        self.transformer2s_3 = nn.ModuleList([nn.Transformer(d_model, 8, 1, 1,  batch_first=True) for _ in range(8)])

        self.embedding1 = nn.Embedding(n_symbols, 80)#160)
        self.text_prenet1 = myPrenet()
        self.prenet1 =  myPrenet()
        self.prenet2 =  myPrenet()
      
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
        y1= self.prenet1(y2)
        y_2= self.prenet2(y2)
        x2 = self.text_prenet1(x2)

        x2_1 = self.positinoal_encoding(x2)
        out_1 = self.positinoal_encoding(y1)
        for transformer in self.transformer1s_3:
            out1 = transformer(x2_1, out_1)
            out_1 = out1 + out_1

        y_2 = self.positinoal_encoding(y_2)
        out_1=self.positinoal_encoding(out_1)
        for transformer in self.transformer2s_3:
            out1 = transformer(y_2, out_1)
            out_1 = out1 + out_1
        mel1 = self.linear_projection1(out_1.permute(0,2,1)).permute(0,2,1)
        return mel1

class Musiclmsum2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm0 = Musiclm0().to(device)
        self.lm1 = Musiclm1().to(device)
    def forward(self, x, y1):   
        out1 = self.lm0(x, y1)
        out2 = self.lm1(x, out1)
        out3 = torch.logaddexp(out1*12, out2*12)/12
        return out1, out2, out3
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_channels, out_channels, stride):
            return nn.Sequential(
                spectral_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(3, 5),
                        stride=(1, stride),
                        padding=(1, 2)
                    )
                ),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.layers = nn.ModuleList([
            block(1, 64, 2),
            block(64, 128, 2),
            block(128, 256, 2),
            block(256, 512, 1),
        ])

        self.output = spectral_norm(
            nn.Conv2d(512, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # [B,T,80] -> [B,1,80,T]
        x = x.permute(0, 2, 1).unsqueeze(1)

        features = []

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        logits = self.output(x)

        return logits, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PatchDiscriminator(),
            PatchDiscriminator(),
            PatchDiscriminator()
        ])

    def forward(self, x):
        outputs = []

        # x: [B,T,80]
        x = x.permute(0, 2, 1).unsqueeze(1)

        for i, discriminator in enumerate(self.discriminators):

            current_x = x.squeeze(1).permute(0, 2, 1)

            logits, features = discriminator(current_x)

            outputs.append((logits, features))

            if i < 2:
                x = F.avg_pool2d(
                    x,
                    kernel_size=(1, 2),
                    stride=(1, 2)
                )

        return outputs


def discriminator_hinge_loss(real_outputs, fake_outputs):

    loss = 0.0

    for (real_logits, _), (fake_logits, _) in zip(
        real_outputs, fake_outputs
    ):

        loss += (
            F.relu(1.0 - real_logits).mean()
            + F.relu(1.0 + fake_logits).mean()
        )

    return loss / len(real_outputs)


def generator_adversarial_loss(fake_outputs):

    loss = 0.0

    for fake_logits, _ in fake_outputs:
        loss += -fake_logits.mean()

    return loss / len(fake_outputs)


def feature_matching_loss(real_outputs, fake_outputs):

    loss = 0.0
    count = 0

    for (real_logits, real_features), (
        fake_logits, fake_features
    ) in zip(real_outputs, fake_outputs):

        for real_feature, fake_feature in zip(
            real_features, fake_features
        ):

            loss += F.l1_loss(
                fake_feature,
                real_feature.detach()
            )

            count += 1

    return loss / max(count, 1)

criterion = MelSpectrogramDetailLoss()

g_model = Musiclmsum2().to(device)
d_model1 = MultiScaleDiscriminator().to(device)
d_model2 = MultiScaleDiscriminator().to(device)
d_model3 = MultiScaleDiscriminator().to(device)
# g_model.load_state_dict(torch.load(f"./pth_save/7g75000.pt", map_location=device))
# d_model1.load_state_dict(torch.load(f"./pth_save/71d75000.pt", map_location=device))
# d_model2.load_state_dict(torch.load(f"./pth_save/72d75000.pt", map_location=device))
# d_model3.load_state_dict(torch.load(f"./pth_save/73d75000.pt", map_location=device))


optimizerG = torch.optim.Adam(
    g_model.parameters(),
    lr=1e-4,
    betas=(0.0, 0.99)
)

optimizerD = torch.optim.Adam(
    list(d_model1.parameters()) + list(d_model2.parameters()) + list(d_model3.parameters()),
    lr=1e-4,
    betas=(0.0, 0.99)
)


LAMBDA_L1 = 40.0
LAMBDA_MSE = 2.0
LAMBDA_FM = 10.0
LAMBDA_ADV = 1.0
LAMBDA_MR = 1.0


while_number =0
encoding_texts_batch =np.load(f"./np_data/encoding_texts_batch.npy")
break_batch = np.load(f"./np_data/break_batch.npy")
accompaniment_batch = np.load(f"./np_data/accompaniment_batch.npy")
# origin_batch = np.load(f"./np_data/origin_batch.npy")
song_batch = np.load(f"./np_data/song_batch.npy")
encoding_texts_batch2 =np.load(f"./np_data_1/encoding_texts_batch.npy")
break_batch2 = np.load(f"./np_data_1/break_batch.npy")
accompaniment_batch2 = np.load(f"./np_data_1/accompaniment_batch.npy")
# origin_batch = np.load(f"./np_data/origin_batch.npy")
song_batch2 = np.load(f"./np_data_1/song_batch.npy")

encoding_texts_batch = np.concatenate([encoding_texts_batch, encoding_texts_batch2], axis=0)
break_batch = np.concatenate([break_batch, break_batch2], axis=0)
accompaniment_batch = np.concatenate([accompaniment_batch, accompaniment_batch2], axis=0)
song_batch = np.concatenate([song_batch, song_batch2], axis=0)

while True:
    
        for epoch in range(0, song_batch.shape[0], 2):
            
                g_model.train()
                d_model1.train()
                d_model2.train()
                d_model3.train()
             
                while_number += 1
                string_data=torch.from_numpy(encoding_texts_batch[epoch:epoch+2]).type(torch.long).to(device)
                breaking_music_data=torch.from_numpy(break_batch[epoch:epoch+2]).type(torch.float).to(device)
                accompaniment_music_data=torch.from_numpy(accompaniment_batch[epoch:epoch+2]).type(torch.float).to(device)
                song_data=torch.from_numpy(song_batch[epoch:epoch+2]).type(torch.float).to(device)


                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=(device == "cuda")
                ):

                    fake_data1, fake_data2, fake_data3= g_model(
                        string_data,
                        breaking_music_data
                    )
                optimizerD.zero_grad(set_to_none=True)

                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=(device == "cuda")
                ):

                    real_outputs1 = d_model1(
                        accompaniment_music_data
                    )
                    real_outputs2 = d_model2(
                        song_data
                    )
                    real_outputs3 = d_model3(
                                            torch.logaddexp(accompaniment_music_data*12, song_data*12)/12
                                        )
               

                    fake_outputs1 = d_model1(
                        fake_data1.detach()
                    )
                    fake_outputs2 = d_model2(
                        fake_data2.detach()
                    )
                    fake_outputs3 = d_model3(
                                            fake_data3.detach()
                                        )
                   

                    loss_D1 = discriminator_hinge_loss(
                        real_outputs1,
                        fake_outputs1
                    )
                    loss_D2 = discriminator_hinge_loss(
                        real_outputs2,
                        fake_outputs2
                    )
                    loss_D3 = discriminator_hinge_loss(
                                            real_outputs3,
                                            fake_outputs3
                                        )
                    
                loss_D = loss_D1+loss_D2+loss_D3
                loss_D.backward()

                optimizerD.step()

                # ==========================================
                # 3. Generator Training
                # ==========================================

                optimizerG.zero_grad(set_to_none=True)

                # 판별기의 가중치는 업데이트하지 않지만
                # Generator까지 전달되는 gradient는 유지한다.
                for parameter in d_model1.parameters():
                    parameter.requires_grad_(False)
                for parameter in d_model2.parameters():
                    parameter.requires_grad_(False)
                for parameter in d_model3.parameters():
                                    parameter.requires_grad_(False)
              
                
                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=(device == "cuda")
                ):

                    # Reconstruction Loss
                    l11 = F.l1_loss(
                        fake_data1,
                        accompaniment_music_data
                    )
                    l12 = F.l1_loss(
                        fake_data2,
                        song_data
                    )
                    l13 = F.l1_loss(
                                            fake_data3,
                                            torch.logaddexp(accompaniment_music_data*12, song_data*12)/12
                                        )                    

                    mse1 = F.mse_loss(
                        fake_data1,
                        accompaniment_music_data
                    )
                    mse2 = F.mse_loss(
                        fake_data2,
                        song_data
                    )
                    mse3 = F.mse_loss(
                                            fake_data3,
                                            torch.logaddexp(accompaniment_music_data*12, song_data*12)/12
                                        )
                   

                    # 시간축 차분
                    diff_time_fake1 = (
                        fake_data1[:, 1:, :]
                        - fake_data1[:, :-1, :]
                    )

                    diff_time_fake2 = (
                        fake_data2[:, 1:, :]
                        - fake_data2[:, :-1, :]
                    )
                    diff_time_fake3 = (
                                            fake_data3[:, 1:, :]
                                            - fake_data3[:, :-1, :]
                                        )

                

                    diff_time_real1 = (
                        accompaniment_music_data[:, 1:, :]
                        - accompaniment_music_data[:, :-1, :]
                    )

                    diff_time_real2 = (
                        song_data[:, 1:, :]
                        - song_data[:, :-1, :]
                    )
                    diff_time_real3 = (
                                            (torch.logaddexp(accompaniment_music_data*12, song_data*12)/12)[:, 1:, :]
                                            - (torch.logaddexp(accompaniment_music_data*12, song_data*12)/12)[:, :-1, :]
                                        )
                  

                    time_grad_loss1 = F.l1_loss(
                        diff_time_fake1,
                        diff_time_real1
                    )

                    time_grad_loss2 = F.l1_loss(
                        diff_time_fake2,
                        diff_time_real2
                    )
                    time_grad_loss3 = F.l1_loss(
                                            diff_time_fake3,
                                            diff_time_real3
                                        )
                    

                    # 주파수축 차분
                    diff_freq_fake1 = (
                        fake_data1[:, :, 1:]
                        - fake_data1[:, :, :-1]
                    )

                    diff_freq_fake2 = (
                        fake_data2[:, :, 1:]
                        - fake_data2[:, :, :-1]
                    )
                    diff_freq_fake3 = (
                                            fake_data3[:, :, 1:]
                                            - fake_data3[:, :, :-1]
                                        )
                  

                    diff_freq_real1 = (
                        accompaniment_music_data[:, :, 1:]
                        - accompaniment_music_data[:, :, :-1]
                    )

                    diff_freq_real2 = (
                        song_data[:, :, 1:]
                        - song_data[:, :, :-1]
                    )
                    diff_freq_real3 = (
                                            (torch.logaddexp(accompaniment_music_data*12, song_data*12)/12)[:, :, 1:]
                                            - (torch.logaddexp(accompaniment_music_data*12, song_data*12)/12)[:, :, :-1]
                                        )

                 

                    freq_grad_loss1 = F.l1_loss(
                        diff_freq_fake1,
                        diff_freq_real1
                    )
                    freq_grad_loss2 = F.l1_loss(
                        diff_freq_fake2,
                        diff_freq_real2
                    )
                    freq_grad_loss3 = F.l1_loss(
                                            diff_freq_fake3,
                                            diff_freq_real3
                                        )

                    # 기존 디테일 손실의 구성 요소
                    loss_recon = (
                        LAMBDA_L1 * l11 +LAMBDA_L1 * l12 +LAMBDA_L1 * l13
                        + LAMBDA_MSE * mse1 + LAMBDA_MSE * mse2 + LAMBDA_MSE * mse3
                        + 0.5 * time_grad_loss1 + 0.5 * time_grad_loss2+ 0.5 * time_grad_loss3
                        + 0.5 * freq_grad_loss1+ 0.5 * freq_grad_loss2+ 0.5 * freq_grad_loss3
                    )

                    # Multi-resolution Mel Loss
                    mr_mel1 = multi_resolution_mel_loss(
                        fake_data1,
                        accompaniment_music_data
                    )
                    mr_mel2 = multi_resolution_mel_loss(
                        fake_data2,
                        song_data
                    )
                    mr_mel3 = multi_resolution_mel_loss(
                                            fake_data3,
                                            torch.logaddexp(accompaniment_music_data*12, song_data*12)/12
                                        )

                    # Generator Adversarial Loss
                    fake_outputs_G1 = d_model1(fake_data1)
                    fake_outputs_G2 = d_model2(fake_data2)
                    fake_outputs_G3 = d_model3(fake_data3)
                   
                    loss_adv1 = generator_adversarial_loss(
                        fake_outputs_G1
                    )
                    loss_adv2 = generator_adversarial_loss(
                        fake_outputs_G2
                    )
                    loss_adv3 = generator_adversarial_loss(
                                            fake_outputs_G3
                                        )
                   

                    # Feature Matching Loss
                    # 실제 데이터 특징은 gradient를 계산하지 않는다.
                    with torch.no_grad():
                        real_outputs_G1 = d_model1(
                            accompaniment_music_data
                        )
                        real_outputs_G2 = d_model2(
                            song_data
                        )
                        real_outputs_G3 = d_model3(
                                                    torch.logaddexp(accompaniment_music_data*12, song_data*12)/12
                                                )
                     

                    loss_fm1 = feature_matching_loss(
                        real_outputs_G1,
                        fake_outputs_G1
                    )

                    loss_fm2 = feature_matching_loss(
                        real_outputs_G2,
                        fake_outputs_G2
                    )
                    loss_fm3 = feature_matching_loss(
                                            real_outputs_G3,
                                            fake_outputs_G3
                                        )
                   
                    # 최종 Generator Loss
                    loss_G = (
                        loss_recon
                        + LAMBDA_MR * mr_mel1+ LAMBDA_MR * mr_mel2+ LAMBDA_MR * mr_mel3
                        + LAMBDA_ADV * loss_adv1+ LAMBDA_ADV * loss_adv2+ LAMBDA_ADV * loss_adv3
                        + LAMBDA_FM * loss_fm1+ LAMBDA_FM * loss_fm2+ LAMBDA_FM * loss_fm3
                    )

                loss_G.backward()

                torch.nn.utils.clip_grad_norm_(
                    g_model.parameters(),
                    max_norm=1.0
                )

                optimizerG.step()

                for parameter in d_model1.parameters():
                    parameter.requires_grad_(True)
                for parameter in d_model2.parameters():
                    parameter.requires_grad_(True)
                for parameter in d_model3.parameters():
                                    parameter.requires_grad_(True)
               

                # ==========================================
                # 4. Logging
                # ==========================================

                print(
                    f"step: {while_number} | "
                    f"D: {loss_D.item():.5f} | "
                    f"G: {loss_G.item():.5f} | "
                    f"Recon: {loss_recon.item():.5f} | "
                    f"MR1: {mr_mel1.item():.5f} | "
                    f"ADV1: {loss_adv1.item():.5f} | "
                    f"FM1: {loss_fm1.item():.5f} | "
                    f"MR2: {mr_mel2.item():.5f} | "
                    f"ADV2: {loss_adv2.item():.5f} | "
                    f"FM2: {loss_fm2.item():.5f} | "
                    f"MR3: {mr_mel3.item():.5f} | "
                    f"ADV3: {loss_adv3.item():.5f} | "
                    f"FM3: {loss_fm3.item():.5f} |"
                 
                )
                
                print(f"step_number : {while_number}, loss_value :    {loss_G.item()}")#  {noize_mse.item()} ")
                if while_number % 101 == 1:
                    g_model.eval()
                    d_model1.eval()
                    d_model2.eval()
                    d_model3.eval()                  
                    with torch.no_grad():       
                        test_num = 0
                     
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            g_fake_data1, g_fake_data2, g_fake_data3 = g_model(string_data[0:1], breaking_music_data[0:1])
                          
                            
                            g_fake_datas1 = (g_fake_data1[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            fake_datas1 = (fake_data1[0].permute(1,0)  * 255).float().cpu().numpy().astype(np.uint8)
                            g_fake_datas2 = (g_fake_data2[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            fake_datas2 = (fake_data2[0].permute(1,0)  * 255).float().cpu().numpy().astype(np.uint8)
                            g_fake_datas3 = (g_fake_data3[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            fake_datas3 = (fake_data3[0].permute(1,0)  * 255).float().cpu().numpy().astype(np.uint8)
                        
                            breaking_music_datas = (breaking_music_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            accompaniment_music_datas = (accompaniment_music_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            song_datas = (song_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            sum_datas = ((torch.logaddexp(accompaniment_music_data[0]*12, song_data[0]*12)/12).permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            
                            img = Image.fromarray(breaking_music_datas)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label1.config(image=photo)
                            img_label1.image = photo 

                            
                            img = Image.fromarray(g_fake_datas1)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label2.config(image=photo)
                            img_label2.image = photo  # 가비지 컬렉션 방
                            
                            img = Image.fromarray(fake_datas1)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label3.config(image=photo)
                            img_label3.image = photo  # 가비지 컬렉션 방지

                            img = Image.fromarray(accompaniment_music_datas)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label4.config(image=photo)
                            img_label4.image = photo 

                            img = Image.fromarray(g_fake_datas2)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label5.config(image=photo)
                            img_label5.image = photo  # 가비지 컬렉션 방
                            
                            img = Image.fromarray(fake_datas2)
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


                            img = Image.fromarray(g_fake_datas3)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label8.config(image=photo)
                            img_label8.image = photo  # 가비지 컬렉션 방
                            
                            img = Image.fromarray(fake_datas3)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label9.config(image=photo)
                            img_label9.image = photo  # 가비지 컬렉션 방지
                            
                            img = Image.fromarray(sum_datas)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 100), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label10.config(image=photo)
                            img_label10.image = photo 
                         
                            root.update()
                
                if while_number % 5000 == 0:
                    torch.save(g_model.state_dict(), f"./pth_save/8g{while_number}.pt")
                    torch.save(d_model1.state_dict(), f"./pth_save/81d{while_number}.pt")
                    torch.save(d_model2.state_dict(), f"./pth_save/82d{while_number}.pt")
                    torch.save(d_model3.state_dict(), f"./pth_save/83d{while_number}.pt")
               
                    

                
