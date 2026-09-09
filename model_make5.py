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

root = tk.Tk()
root.title("5")
img_label1 = tk.Label(root, width=1300, height=50, bg="black")
img_label2 = tk.Label(root, width=1300, height=50, bg="black")
img_label3 = tk.Label(root, width=1300, height=50, bg="black")
img_label4 = tk.Label(root, width=1300, height=50, bg="black")
img_label5 = tk.Label(root, width=1300, height=50, bg="black")
img_label6 = tk.Label(root, width=1300, height=50, bg="black")
img_label7 = tk.Label(root, width=1300, height=50, bg="black")
img_label8 = tk.Label(root, width=1300, height=50, bg="black")
img_label9 = tk.Label(root, width=1300, height=50, bg="black")
img_label10 = tk.Label(root, width=1300, height=50, bg="black")

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

class myPrenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(80, 160, 4, 2, 1),
            nn.BatchNorm1d(160),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Conv1d(160, 320, 4, 2, 1),
            nn.BatchNorm1d(320),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Conv1d(320, 512, 4, 2, 1),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Conv1d(512, 512, 4, 2, 1),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.LeakyReLU(),            
        )
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0,2,1)
        return x
class Musiclm2(nn.Module):
    def __init__(self):
        super().__init__()
        self.charset = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄶㄳㄵㄺㄼㄽㄾㄿㅀㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣabcdefghijklmnopqrstuvwxyz1234567890`'\"\\?/><.,!()~@#$%^&*-_+= \n\r"    
        n_symbols = len(self.charset) + 2
        d_model =512
        n_mel_channels = 80
        self.n_mel_channels = n_mel_channels
        self.positinoal_encoding = PositionalEncoding(d_model, max_len=500*3)  
        #8000 4000 2000 1000 500

        self.transformer4s_1 = nn.ModuleList([nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model*4, batch_first=True), 1) for _ in range(10)])  
        self.transformer4s_2 = nn.ModuleList([nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model*4, batch_first=True), 1) for _ in range(10)])     
        #self.transformer4s_2 = nn.ModuleList([nn.Transformer(d_model, 8,1, 1, dim_feedforward=d_model*4, batch_first=True, dropout=0) for _ in range(10)]) 
        # self.transformer4_3 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model*4, batch_first=True),1)
        # self.transformer4_4 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model*4, batch_first=True),1)
        # self.transformer4_5 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model*4, batch_first=True),1)
        

        self.embedding2 = nn.Embedding(n_symbols, 80)#160)
        self.embedding3 = nn.Embedding(n_symbols, 80)#160)
        self.text_prenet2 = myPrenet()
        self.text_prenet3 = myPrenet()
        self.prenet2 =  myPrenet()
        self.prenet3 =  myPrenet()
        self.linear_projection2 = nn.Sequential(
            nn.ConvTranspose1d(512, 512, 4, 2, 1),
            nn.BatchNorm1d(512),
            # nn.Dropout(),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 320, 4, 2, 1),
            nn.BatchNorm1d(320),
            # nn.Dropout(),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(320, 160, 4, 2, 1),
            nn.BatchNorm1d(160),
            # nn.Dropout(),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(160, 80, 4, 2, 1) 
        )
        self.linear_projection3 = nn.Sequential(
            nn.ConvTranspose1d(512, 512, 4, 2, 1),
            nn.BatchNorm1d(512),
            # nn.Dropout(),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 320, 4, 2, 1),
            nn.BatchNorm1d(320),
            # nn.Dropout(),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(320, 160, 4, 2, 1),
            nn.BatchNorm1d(160),
            # nn.Dropout(),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(160, 80, 4, 2, 1) 
        )
                        
        
    def forward(self, x, y1):
        x2 = self.embedding2(x)
        x2 = self.text_prenet2(x2)
    
        y= self.prenet2(y1)
        x_2_1 = x2
        y_2_1 = y
        for transformer in self.transformer4s_1:
            y_2 = y_2_1
            y_2 = transformer(self.positinoal_encoding(torch.cat([x_2_1, y_2], dim=1)))[:, -y_2.shape[1]:, :]
            y_2_1 = y_2 + y_2_1

     
        mel2 = self.linear_projection2(y_2_1.permute(0,2,1)).permute(0,2,1)

        x2 = self.embedding3(x)
        x2 = self.text_prenet3(x2)
        
        y= self.prenet3(mel2)
        x_2_1 = x2
        y_2_1 = y
        for transformer in self.transformer4s_2:
            x_2 = x_2_1
            x_2 = transformer(self.positinoal_encoding(torch.cat([y_2_1, x_2], dim=1)))[:, -x_2.shape[1]:, :]
            x_2_1 = x_2 + x_2_1
        
             
        mel3 = self.linear_projection3(x_2_1.permute(0,2,1)).permute(0,2,1)


        sum_mel = torch.logaddexp(mel2*12, mel3*12)/12
        return mel2, mel3, sum_mel

class MelLoss(nn.Module):
    def __init__(
        self,
        l1_weight=1.0,
        l2_weight=0.5,
        time_weight=0.5,
        freq_weight=0.5,
        sc_weight=0.1,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.time_weight = time_weight
        self.freq_weight = freq_weight
        self.sc_weight = sc_weight

    def forward(self, pred, target):
        """
        pred   : [B, n_mels, T] 또는 [B, T, n_mels]
        target : pred와 동일한 shape

        이미 log-mel인 경우를 가정.
        """

        # -------------------------------------------------
        # 1. 기본 L1 loss
        # -------------------------------------------------
        l1 = F.l1_loss(pred, target)

        # -------------------------------------------------
        # 2. L2 loss
        # -------------------------------------------------
        l2 = F.mse_loss(pred, target)

        # -------------------------------------------------
        # 3. 시간축 변화량 loss
        # -------------------------------------------------
        pred_time = pred[..., 1:] - pred[..., :-1]
        target_time = target[..., 1:] - target[..., :-1]

        time_loss = F.l1_loss(pred_time, target_time)

        # -------------------------------------------------
        # 4. 주파수축 변화량 loss
        # -------------------------------------------------
        pred_freq = pred[:, 1:, ...] - pred[:, :-1, ...]
        target_freq = target[:, 1:, ...] - target[:, :-1, ...]

        freq_loss = F.l1_loss(pred_freq, target_freq)

        # -------------------------------------------------
        # 5. Spectral Convergence
        # -------------------------------------------------
        diff = pred - target

        numerator = torch.linalg.vector_norm(
            diff.reshape(diff.shape[0], -1),
            dim=1
        )

        denominator = torch.linalg.vector_norm(
            target.reshape(target.shape[0], -1),
            dim=1
        )

        sc = (numerator / (denominator + 1e-8)).mean()

        # -------------------------------------------------
        # 최종 loss
        # -------------------------------------------------
        loss = (
            self.l1_weight * l1
            + self.l2_weight * l2
            + self.time_weight * time_loss
            + self.freq_weight * freq_loss
            + self.sc_weight * sc
        )

        return loss
    
g_model1 = Musiclm2().to(device)
# d_model1 = MultiScaleDiscriminator().to(device)
# d_model2 = MultiScaleDiscriminator().to(device)
# d_model3 = MultiScaleDiscriminator().to(device)
g_model1 = torch.load(
   "./pth_save/2g590000.pt",
   weights_only=False,
)



optimizerG = torch.optim.Adam(g_model1.parameters(), lr=1e-4)#, betas=(0.0, 0.99))
# optimizerD = torch.optim.Adam(
#         list(d_model1.parameters()) + list(d_model2.parameters()) + list(d_model3.parameters()), lr=1e-4, betas=(0.0, 0.99))

criterion_gan = MelLoss() # LSGAN 손실함수 주로 사용
while_number = 590000
encoding_texts_batch =np.load(f"./np_data/encoding_texts_batch.npy")
break_batch = np.load(f"./np_data/break_batch.npy")
accompaniment_batch = np.load(f"./np_data/accompaniment_batch.npy")
origin_batch = np.load(f"./np_data/origin_batch.npy")
song_batch = np.load(f"./np_data/song_batch.npy")

scalerG = torch.amp.GradScaler("cuda")

while True:
    
        for epoch in range(0, song_batch.shape[0], 5):
            
                g_model1.train()
                # d_model1.train()
                # d_model2.train()
                # d_model3.train()
                
                ##-12. 15
                while_number += 1
                string_data=torch.from_numpy(encoding_texts_batch[epoch:epoch+5]).type(torch.long).to(device)
                breaking_music_data=torch.from_numpy(break_batch[epoch:epoch+5]).type(torch.float).to(device)
                accompaniment_music_data=torch.from_numpy(accompaniment_batch[epoch:epoch+5]).type(torch.float).to(device)
                song_data=torch.from_numpy(song_batch[epoch:epoch+5]).type(torch.float).to(device)
           
           
                fake_data3, fake_data4,output_real2,output_real_feat2, output_fake3, output_fake4,output_fake_for_G3, output_fake_for_G_feat3, output_fake_for_G4, output_fake_for_G_feat4=None, None, None, None, None, None, None, None, None, None
                # with torch.autocast("cuda", dtype=torch.bfloat16):
                #     output_real3s = d_model1(accompaniment_music_data)
                #     output_real4s = d_model2(song_data)
                #     output_real5s = d_model3(torch.logaddexp(accompaniment_music_data*12, song_data*12)/12)
                
                #     fake_data3, fake_data4, fake_data5 = g_model1.forward(string_data, breaking_music_data) 
                #     output_fake3s = d_model1(fake_data3.detach()) 
                #     output_fake4s = d_model2(fake_data4.detach())
                #     output_fake5s = d_model3(fake_data5.detach())
                #     loss_D_3, loss_D_4, loss_D_5 = 0.0, 0.0, 0.0
                #     for output_real3, output_fake3 in zip(output_real3s, output_fake3s):
                #         loss_D_3 += criterion_gan(output_real3, torch.ones_like(output_real3)) + criterion_gan(output_fake3, torch.zeros_like(output_fake3))
                #     for output_real4, output_fake4 in zip(output_real4s, output_fake4s):
                #         loss_D_4 += criterion_gan(output_real4, torch.ones_like(output_real4)) + criterion_gan(output_fake4, torch.zeros_like(output_fake4))
                #     for output_real5, output_fake5 in zip(output_real5s, output_fake5s):
                #         loss_D_5 += criterion_gan(output_real5, torch.ones_like(output_real5)) + criterion_gan(output_fake5, torch.zeros_like(output_fake5))
                #     loss_D = loss_D_3 + loss_D_4 + loss_D_5
            
                # optimizerD.zero_grad()
                # loss_D.backward()
                # nn.utils.clip_grad_norm_(d_model1.parameters(), max_norm=1.0)
                # nn.utils.clip_grad_norm_(d_model2.parameters(), max_norm=1.0)
                # nn.utils.clip_grad_norm_(d_model3.parameters(), max_norm=1.0)
                # nn.utils.clip_grad_norm_(g_model1.parameters(), max_norm=1.0)
                # optimizerD.step()
            
                before_params = {
                    name: param.detach().clone()
                    for name, param in g_model1.named_parameters()
                    if param.requires_grad
                }

     
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fake_data3, fake_data4, fake_data5= g_model1.forward(string_data, breaking_music_data) 

                    # output_real3s = d_model1(accompaniment_music_data)
                    # output_real4s = d_model2(song_data)
                    # output_real5s = d_model3(torch.logaddexp(accompaniment_music_data*12, song_data*12)/12)

                    # output_fake_for_G3s = d_model1(fake_data3)      
                    # output_fake_for_G4s = d_model2(fake_data4)
                    # output_fake_for_G5s = d_model3(fake_data5)

                    # loss_l3, loss_l4, loss_l5 = 0.0, 0.0, 0.0
                    # for output_fake_for_G3 in output_fake_for_G3s:
                    #     loss_l3 += criterion_gan(output_fake_for_G3, torch.ones_like(output_fake_for_G3))
                    loss_l3 = criterion_gan(fake_data3, accompaniment_music_data)
                    # for output_fake_for_G4 in output_fake_for_G4s:
                    #     loss_l4 += criterion_gan(output_fake_for_G4, torch.ones_like(output_fake_for_G4))
                    loss_l4 = criterion_gan(fake_data4, song_data)
                    # for output_fake_for_G5 in output_fake_for_G5s:
                    #     loss_l5 += criterion_gan(output_fake_for_G5, torch.ones_like(output_fake_for_G5))
                    loss_l5 = criterion_gan(fake_data5, torch.logaddexp(accompaniment_music_data*12, song_data*12)/12)
                    loss_G = loss_l3 + loss_l4 + loss_l5

                optimizerG.zero_grad()
                loss_G.backward()
                # nn.utils.clip_grad_norm_(d_model1.parameters(), max_norm=1.0)
                # nn.utils.clip_grad_norm_(d_model2.parameters(), max_norm=1.0)
                # nn.utils.clip_grad_norm_(d_model3.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(g_model1.parameters(), max_norm=1.0)                 
                optimizerG.step()

                grad_sum = 0.0
                grad_count = 0
                grad_max = 0.0

                for name, param in g_model1.named_parameters():

                    if param.grad is not None:

                        grad = param.grad.detach().abs()

                        grad_mean = grad.mean().item()
                        grad_max_value = grad.max().item()

                        grad_sum += grad_mean
                        grad_count += 1

                        grad_max = max(
                            grad_max,
                            grad_max_value
                        )

                
                parameter_change = 0.0
                max_parameter_change = 0.0
                changed_params = 0

                for name, param in g_model1.named_parameters():

                    if param.requires_grad:

                        change = (
                            param.detach() - before_params[name]
                        ).abs()

                        mean_change = change.mean().item()
                        max_change = change.max().item()

                        parameter_change += mean_change

                        max_parameter_change = max(
                            max_parameter_change,
                            max_change
                        )

                        if mean_change > 0:
                            changed_params += 1


                # ============================================
                # 학습 상태 자동 판단
                # ============================================
                print(optimizerG.param_groups[0]["lr"])
                if grad_count == 0:

                    status = "❌ Gradient 없음 → 계산 그래프 단절 가능성"

                elif grad_sum == 0:

                    status = "❌ Gradient = 0 → 학습이 진행되지 않을 가능성"

                elif parameter_change == 0:

                    status = "❌ Parameter 변화 없음 → optimizer 업데이트 문제 확인"

                elif parameter_change < 1e-10:

                    status = "⚠️ Parameter 변화가 극도로 작음 → 학습 정체 가능성"

                else:

                    status = "✅ 모델 Parameter 업데이트 중 → 학습 진행"


                # ============================================
                # 출력
                # ============================================
                print("=" * 70)

                print(f"Gradient Mean Sum     : {grad_sum:.12e}")

                print(f"Gradient Max          : {grad_max:.12e}")

                print(f"Parameters With Grad  : {grad_count}")

                print(f"Changed Parameters    : {changed_params}")

                print(f"Parameter Change      : {parameter_change:.12e}")

                print(f"Max Parameter Change  : {max_parameter_change:.12e}")

                print(f"Learning Status       : {status}")

                print("=" * 70)
                print(f"step_number : {while_number}, loss_value : {loss_G.item()}")
                if while_number % 101 == 1:
                    g_model1.eval()
                    # d_model1.eval()
                    # d_model2.eval()
                    # d_model3.eval()
                    with torch.no_grad():       
                        test_num = 0
                        # with sdpa_kernel(
                        #     [SDPBackend.FLASH_ATTENTION]
                        # ):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            g_fake_data3, g_fake_data4, g_fake_data5 = g_model1.forward(string_data[0:1], breaking_music_data[0:1])
                                        
                           
                            g_fake_datas3 = (g_fake_data3[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            g_fake_datas4 = (g_fake_data4[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            g_fake_datas5 = (g_fake_data5[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)

                           
                            fake_datas3 = (fake_data3[0].permute(1,0)  * 255).float().cpu().numpy().astype(np.uint8)
                            fake_datas4 = (fake_data4[0].permute(1,0)  * 255).float().cpu().numpy().astype(np.uint8)
                            fake_datas5 = (fake_data5[0].permute(1,0)  * 255).float().cpu().numpy().astype(np.uint8)
                            breaking_music_datas = (breaking_music_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            accompaniment_music_datas = (accompaniment_music_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            song_datas = (song_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            # 1. 분자 분모 모두 GPU(또는 현재 디바이스)에서 연산 진행
                            numerator = (torch.logaddexp(accompaniment_music_data[0]*12, song_data[0]*12)/12).permute(1, 0).float()
                            original_music_datas = (numerator* 255).cpu().numpy().astype(np.uint8)

                            img = Image.fromarray(breaking_music_datas)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label1.config(image=photo)
                            img_label1.image = photo 
                        
                            img = Image.fromarray(g_fake_datas3)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label2.config(image=photo)
                            img_label2.image = photo  # 가비지 컬렉션 방
                            
                            img = Image.fromarray(fake_datas3)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label3.config(image=photo)
                            img_label3.image = photo  # 가비지 컬렉션 방지

                            img = Image.fromarray(accompaniment_music_datas)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label4.config(image=photo)
                            img_label4.image = photo 

                            img = Image.fromarray(g_fake_datas4)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label5.config(image=photo)
                            img_label5.image = photo  # 가비지 컬렉션 방
                            
                            img = Image.fromarray(fake_datas4)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label6.config(image=photo)
                            img_label6.image = photo  # 가비지 컬렉션 방지

                            img = Image.fromarray(song_datas)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label7.config(image=photo)
                            img_label7.image = photo 

                            img = Image.fromarray(g_fake_datas5)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label8.config(image=photo)
                            img_label8.image = photo  # 가비지 컬렉션 방
                            
                            img = Image.fromarray(fake_datas5)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label9.config(image=photo)
                            img_label9.image = photo  # 가비지 컬렉션 방지

                            img = Image.fromarray(original_music_datas)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label10.config(image=photo)
                            img_label10.image = photo 
                        

            
                            root.update()
                
                if while_number % 5000 == 0:
                    torch.save(g_model1, f"./pth_save/1g{while_number}.pt")
                    # torch.save(d_model1, f"./pth_save/1d1{while_number}.pt")
                    # torch.save(d_model2, f"./pth_save/1d2{while_number}.pt")
                    # torch.save(d_model3, f"./pth_save/1d3{while_number}.pt")
                    

                