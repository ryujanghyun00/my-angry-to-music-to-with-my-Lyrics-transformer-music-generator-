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
root.title("5_1")
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

# img_label1.grid(row=0, column=0)
# img_label2.grid(row=1, column=0)
# img_label3.grid(row=0, column=0)
img_label4.grid(row=0, column=0)
img_label5.grid(row=1, column=0)
img_label6.grid(row=2, column=0)
img_label7.grid(row=3, column=0)
# img_label8.grid(row=7, column=0)
# img_label9.grid(row=8, column=0)
# img_label10.grid(row=9, column=0)

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

        self.transformer4s_1 = nn.MultiheadAttention(d_model, 32, batch_first=True)
        self.transformer4s_2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 32,dropout=0.0, batch_first=True), 4)
      
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

        x2, _ = self.transformer4s_1(y, x2, x2)
        out = self.transformer4s_2(x2)
        mel3 = self.linear_projection3(out.permute(0,2,1)).permute(0,2,1)
        return mel3
    

class PatchDiscriminator1D(nn.Module):
    """1D PatchGAN that judges local temporal/spectral patterns in a mel."""
    def __init__(self, in_channels=N_MELS, base=64):
        super().__init__()
        layers = []
        channels = [in_channels, base, base * 2, base * 4, base * 4]
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=5,
                          stride=2 if i < 3 else 1, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers.append(nn.Conv1d(channels[-1], 1, kernel_size=3, padding=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                features.append(x)
        return x, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PatchDiscriminator1D() for _ in range(scales)]
        )

    def forward(self, mel):
        scores, all_features = [], []
        x = mel
        for i, disc in enumerate(self.discriminators):
            score, features = disc(x)
            scores.append(score)
            all_features.append(features)
            if i != len(self.discriminators) - 1:
                x = F.avg_pool1d(x, kernel_size=4, stride=2, padding=1)
        return scores, all_features


# ----------------------------- Loss functions ----------------------------

def reconstruction_losses(fake, real):
    """L1 + MSE on the same mel scale."""
    l1 = F.l1_loss(fake, real)
    mse = F.mse_loss(fake, real)
    return l1, mse


def discriminator_hinge_loss(real_scores, fake_scores):
    loss = 0.0
    for real_s, fake_s in zip(real_scores, fake_scores):
        loss = loss + (F.relu(1.0 - real_s).mean() +
                       F.relu(1.0 + fake_s).mean())
    return loss / len(real_scores)


def generator_hinge_loss(fake_scores):
    return sum(-score.mean() for score in fake_scores) / len(fake_scores)


def feature_matching_loss(real_features, fake_features):
    """
    Stabilizes GAN training by matching intermediate discriminator features.
    Real features are detached so this loss does not update the discriminator.
    """
    total = 0.0
    count = 0
    for real_scale, fake_scale in zip(real_features, fake_features):
        for real_layer, fake_layer in zip(real_scale, fake_scale):
            total = total + F.l1_loss(fake_layer, real_layer.detach())
            count += 1
    return total / max(count, 1)


def multi_resolution_mel_loss(fake, real):
    """
    Simple multi-resolution mel loss: compare at full, half, and quarter time
    resolutions. This is NOT a waveform STFT loss; it is appropriate as an
    additional mel-domain consistency term.
    """
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

g_model1 = Musiclm1().to(device)
# d_model1 = GanModel().to(device)
d_model2 = MultiScaleDiscriminator(scales=3).to(device)
# d_model3 = GanModel().to(device)
# d_model3 = MultiScaleDiscriminator().to(device)
# g_model1 = torch.load(
#    "./pth_save/1g120000.pt",
#    weights_only=False,
# )

# d_model1 = torch.load(
#    "./pth_save/1d1120000.pt",
#    weights_only=False,
# )

# d_model2 = torch.load(
#    "./pth_save/1d2120000.pt",
#    weights_only=False,
# )


optimizerG = torch.optim.Adam(g_model1.parameters(), lr=1e-4, betas=(0.0, 0.99))
optimizerD = torch.optim.Adam(
        list(d_model2.parameters()), lr=1e-4, betas=(0.0, 0.99))

criterion_gan = nn.BCEWithLogitsLoss() # LSGAN 손실함수 주로 사용
while_number = 0
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
                d_model2.train()
                # d_model3.train()
                
                ##-12. 15
                while_number += 1
                string_data=torch.from_numpy(encoding_texts_batch[epoch:epoch+5]).type(torch.long).to(device)
                breaking_music_data=torch.from_numpy(break_batch[epoch:epoch+5]).type(torch.float).to(device)
                accompaniment_music_data=torch.from_numpy(accompaniment_batch[epoch:epoch+5]).type(torch.float).to(device)
                song_data=torch.from_numpy(song_batch[epoch:epoch+5]).type(torch.float).to(device)
           
           
                fake_data3, fake_data4,output_real2,output_real_feat2, output_fake3, output_fake4,output_fake_for_G3, output_fake_for_G_feat3, output_fake_for_G4, output_fake_for_G_feat4=None, None, None, None, None, None, None, None, None, None
                # with torch.autocast("cuda", dtype=torch.bfloat16):
                # output_real3 = d_model1(accompaniment_music_data)
                output_real4, _ = d_model2(song_data.permute(0,2,1))
                # output_real5 = d_model3(torch.logaddexp(accompaniment_music_data*12, song_data*12)/12)
            
                fake_data4 = g_model1.forward(string_data, accompaniment_music_data) 
                # output_fake3 = d_model1(fake_data3.detach()) 
                output_fake4, _ = d_model2(fake_data4.detach().permute(0,2,1))
                # output_fake5 = d_model3(fake_data5.detach())

                # loss_D_3 = criterion_gan(output_real3, torch.ones_like(output_real3)) + criterion_gan(output_fake3, torch.zeros_like(output_fake3))
                loss_d = discriminator_hinge_loss(output_real4, output_fake4)
                # loss_D_5 = criterion_gan(output_real5, torch.ones_like(output_real5)) + criterion_gan(output_fake5, torch.zeros_like(output_fake5))
                loss_D =loss_d #+ loss_D_5
            
                optimizerD.zero_grad()
                loss_D.backward()
                # nn.utils.clip_grad_norm_(d_model1.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(d_model2.parameters(), max_norm=5.0)
                optimizerD.step()
            
                before_params = {
                    name: param.detach().clone()
                    for name, param in g_model1.named_parameters()
                    if param.requires_grad
                }

     
                # with torch.autocast("cuda", dtype=torch.bfloat16):
                fake_data4 = g_model1.forward(string_data, accompaniment_music_data) 

                # output_real3 = d_model1(accompaniment_music_data)
                output_real4, real_features = d_model2(song_data.permute(0,2,1))
                # output_real5 = d_model3(torch.logaddexp(accompaniment_music_data*12, song_data*12)/12)

                # output_fake_for_G3 = d_model1(fake_data3)      
                output_fake_for_G4, fake_features = d_model2(fake_data4.permute(0,2,1))
                # output_fake_for_G5 = d_model3(fake_data5)

                l1, mse = reconstruction_losses(fake_data4, song_data)
                mr_mel = multi_resolution_mel_loss(fake_data4, song_data)
                adv = generator_hinge_loss(output_fake_for_G4)
                fm = feature_matching_loss(real_features, fake_features)
            
                loss_G = loss_g = (
                           LAMBDA_L1 * l1
                           + LAMBDA_MSE * mse
                           + 5.0 * mr_mel
                           + LAMBDA_ADV * adv
                           + LAMBDA_FM * fm
                       )
                optimizerG.zero_grad()
                loss_G.backward()
                # nn.utils.clip_grad_norm_(d_model1.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(g_model1.parameters(), max_norm=5.0)                 
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
                print(f"step_number : {while_number}, loss_value :    {loss_G.item()}   ")
                if while_number % 101 == 1:
                    g_model1.eval()
                    # d_model1.eval()
                    d_model2.eval()
                    # d_model3.eval()
                    with torch.no_grad():       
                        test_num = 0
                        # with sdpa_kernel(
                        #     [SDPBackend.FLASH_ATTENTION]
                        # ):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            g_fake_data4 = g_model1(string_data[0:1], accompaniment_music_data[0:1])
                                        
                           
                            # g_fake_datas3 = (g_fake_data3[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            g_fake_datas4 = (g_fake_data4[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            # g_fake_datas5 = (g_fake_data5[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)

                           
                            # fake_datas3 = (fake_data3[0].permute(1,0)  * 255).float().cpu().numpy().astype(np.uint8)
                            fake_datas4 = (fake_data4[0].permute(1,0)  * 255).float().cpu().numpy().astype(np.uint8)
                            # fake_datas5 = (fake_data5[0].permute(1,0)  * 255).float().cpu().numpy().astype(np.uint8)
                            # breaking_music_datas = (breaking_music_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            accompaniment_music_datas = (accompaniment_music_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            song_datas = (song_data[0].permute(1,0) * 255).float().cpu().numpy().astype(np.uint8)
                            # 1. 분자 분모 모두 GPU(또는 현재 디바이스)에서 연산 진행
                            # numerator = (torch.logaddexp(accompaniment_music_data[0]*12, song_data[0]*12)/12).permute(1, 0).float()
                            # original_music_datas = (numerator* 255).cpu().numpy().astype(np.uint8)

                            # img = Image.fromarray(breaking_music_datas)
                            # img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            # img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            # photo = ImageTk.PhotoImage(img)
                            # img_label1.config(image=photo)
                            # img_label1.image = photo 
                        
                            # img = Image.fromarray(g_fake_datas3)
                            # img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            # img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            # photo = ImageTk.PhotoImage(img)
                            # img_label2.config(image=photo)
                            # img_label2.image = photo  # 가비지 컬렉션 방
                            
                            # img = Image.fromarray(fake_datas3)
                            # img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            # img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            # photo = ImageTk.PhotoImage(img)
                            # img_label3.config(image=photo)
                            # img_label3.image = photo  # 가비지 컬렉션 방지

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

                            # img = Image.fromarray(g_fake_datas5)
                            # img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            # img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            # photo = ImageTk.PhotoImage(img)
                            # img_label8.config(image=photo)
                            # img_label8.image = photo  # 가비지 컬렉션 방
                            
                            # img = Image.fromarray(fake_datas5)
                            # img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            # img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            # photo = ImageTk.PhotoImage(img)
                            # img_label9.config(image=photo)
                            # img_label9.image = photo  # 가비지 컬렉션 방지

                            # img = Image.fromarray(original_music_datas)
                            # img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                            # img = img.resize((1300, 50), Image.Resampling.LANCZOS)
                            # photo = ImageTk.PhotoImage(img)
                            # img_label10.config(image=photo)
                            # img_label10.image = photo 
                        

            
                            root.update()
                
                if while_number % 5000 == 0:
                    torch.save(g_model1, f"./pth_save/1g{while_number}.pt")
                    # torch.save(d_model1, f"./pth_save/1d1{while_number}.pt")
                    torch.save(d_model2, f"./pth_save/1d{while_number}.pt")
                    # torch.save(d_model3, f"./pth_save/1d3{while_number}.pt")
                    

                
