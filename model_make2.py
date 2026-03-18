import torch
import torch.nn as nn
# from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
import math
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000)/d_model))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:,:x.size(1)]
    
class MelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Up(1024,1024, 100),
            Up(1024,1024, 2000),
            Up(1024,512, 4000),
            Up(512,256, 8000),
            Up(256,128, 16000),
        )

    def forward(self,x):
        x = self.net(x)
        return x
class textCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Down(128,256, 2000), #4000 -> 2000 1000 -> 500 250
            Down(256,512, 1000),
            Down(512,1024, 500),
        )

    def forward(self,x):
        # x (B,L,E)
        x = self.net(x)
        return x

class Up(nn.Module):
    def __init__(self,in_ch,out_ch, lens):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch,out_ch,4,2,1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
    def forward(self,x):
        return self.block(x)
class MelCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Down(128,256, 8000),  #16000 -> 8000 4000 -> 2000 1000 -> 500 250
            Down(256,512, 4000),
            Down(512,1024, 2000),
            Down(1024,1024, 1000),
            Down(1024,1024, 500),
        )

    def forward(self,x):
        # x (B,L,E)
        x = self.net(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, lens):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch,out_ch,7,2,3),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
    def forward(self,x):
        return self.block(x)

class Ganerator(nn.Module):
    def __init__(self, ngf=128):
        super(Ganerator, self).__init__()
        self.charset = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄶㄳㄵㄺㄼㄽㄾㄿㅀㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣabcdefghijklmnopqrstuvwxyz1234567890`'\"\\?/><.,!()~@#$%^&*-_+= \n\r"    
        self.embedding = nn.Embedding(self.charset.__len__()+2, 128)  
        self.p_enc_1d_model_string_sum = PositionalEncoding(1024, max_len=500)
        self.p_enc_1d_model_music_sum = PositionalEncoding(1024, max_len=500)
        self.p_enc_1d_medel_output_sum = PositionalEncoding(1024, max_len=500)


        self.transformer_music = nn.Transformer(1024,  num_encoder_layers = 3, num_decoder_layers = 3, batch_first=True)
        self.transformer_music2 = nn.Transformer(1024,  num_encoder_layers = 3, num_decoder_layers = 3, batch_first=True)
        
        self.transformer_gasa = nn.Transformer(1024,  num_encoder_layers = 3, num_decoder_layers = 3, batch_first=True)

        self.transformer_song = nn.Transformer(1024,  num_encoder_layers = 3, num_decoder_layers = 3, batch_first=True)


        self.sequnce_text = textCompressor()

        self.sequnce_audio = MelCompressor()

        self.sequnce_audio2 = MelCompressor()

        self.unsequnce_aduio = nn.Sequential(
            MelDecoder(),
            nn.Conv1d(128, 128, 3, 1, 1),
        )

        self.unsequnce_aduio2 = nn.Sequential(
            MelDecoder(),
            nn.Conv1d(128, 128, 3, 1, 1),
        )
        # torch.Size([15, 4000]) torch.Size([15, 16000, 128])
            
    def forward(self, string_data, breaking_music_data):
        embeded_string_data=self.embedding(string_data) #batch, x, ch

        embeded_string_data = embeded_string_data.permute(0, 2, 1)
        embeded_string_data = self.sequnce_text(embeded_string_data)
        embeded_string_data = embeded_string_data.permute(0, 2, 1)

        breaking_music_data = breaking_music_data.permute(0, 2, 1)
        breaking_music_data = self.sequnce_audio(breaking_music_data)
        breaking_music_data = breaking_music_data.permute(0, 2, 1)

        embeded_string_data = self.p_enc_1d_model_string_sum(embeded_string_data)
        
        output1_1 = self.p_enc_1d_model_music_sum(breaking_music_data)
        output1_1 = self.transformer_music(embeded_string_data, output1_1)
        output1_1 = self.transformer_music2(output1_1, output1_1)
        
        output1_1 = self.unsequnce_aduio(output1_1.permute(0, 2, 1))
        output1_1 = output1_1.permute(0, 2, 1)
        
        output1 = output1_1.permute(0, 2, 1)
        output1 = self.sequnce_audio2(output1)
        output1 = output1.permute(0, 2, 1)
        
        output1 = self.p_enc_1d_medel_output_sum(output1)

        output1 = self.transformer_song(output1, embeded_string_data)
        output1 = self.transformer_gasa(output1, output1)

        output1 = output1.permute(0,2,1)
        output1 = self.unsequnce_aduio2(output1)
        output1 = output1.permute(0,2,1)

        sum_song = torch.logsumexp(torch.stack([output1, output1_1], dim=0), dim=0)
        
        return  output1, output1_1, sum_song
    # output,
class Discreminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = nn.Conv1d(128, 256, 4,2,1)
        self.y1 = nn.BatchNorm1d(256)
        self.z1 = nn.GELU()
        
        self.x2 = nn.Conv1d(256, 512, 4,2,1)
        self.y2 = nn.BatchNorm1d(512)
        self.z2 = nn.GELU()
        self.x3 = nn.Conv1d(512, 1024, 4,2,1)
        self.y3 = nn.BatchNorm1d(1024)
        self.z3 = nn.GELU()
        
        self.output = nn.Conv1d(1024, 1, 3, 1, 1)
        
    def forward(self, data):
        #data 16000 128
        # print(data.shape)
        data = data.permute(0, 2, 1)
        feats = []
        x = self.z1(self.y1(self.x1(data)))
        feats.append(x)

        x = self.z2(self.y2(self.x2(x)))
        feats.append(x)

        x = self.z3(self.y3(self.x3(x)))
        feats.append(x)

        data = self.output(x)
        
        return data, feats

def discriminator_loss(real_out, fake_out):
    loss_real = torch.mean(F.relu(1 - real_out))
    loss_fake = torch.mean(F.relu(1 + fake_out))
    return loss_real + loss_fake

def generator_loss(fake_out):
    return -torch.mean(fake_out)
def feature_matching_loss(real_feats, fake_feats):

    loss = 0

    for real, fake in zip(real_feats, fake_feats):
        loss += torch.mean(torch.abs(real - fake))

    return loss
def mel_recon_loss(real_mel, fake_mel):

    return torch.mean(torch.abs(real_mel - fake_mel))
def generator_total_loss(fake_out, real_mel, fake_mel, real_feats, fake_feats):

    loss_gan = generator_loss(fake_out)

    loss_mel = mel_recon_loss(real_mel, fake_mel)

    loss_fm = feature_matching_loss(real_feats, fake_feats)

    loss = (
        loss_gan
        + 35 * loss_mel
        + 2 * loss_fm
    )

    return loss
def discriminator_total_loss(real_out, fake_out):

    loss_d = discriminator_loss(real_out, fake_out)

    return loss_d

root = tk.Tk()
root.title("2")
img_label1 = tk.Label(root, width=1800, height=70, bg="black")
img_label2 = tk.Label(root, width=1800, height=70, bg="black")
img_label3 = tk.Label(root, width=1800, height=70, bg="black")
img_label4 = tk.Label(root, width=1800, height=70, bg="black")
img_label5 = tk.Label(root, width=1800, height=70, bg="black")
img_label6 = tk.Label(root, width=1800, height=70, bg="black")
img_label7 = tk.Label(root, width=1800, height=70, bg="black")
img_label8 = tk.Label(root, width=1800, height=70, bg="black")
img_label9 = tk.Label(root, width=1800, height=70, bg="black")
img_label10 = tk.Label(root, width=1800, height=70, bg="black")
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


g_model = Ganerator().to(device)
d_model = Discreminator().to(device)
d_model_2 = Discreminator().to(device)
d_model_3 = Discreminator().to(device)
# g_model = torch.load('./pth_save/1g30000.pt', weights_only=False)
# d_model = torch.load('./pth_save/1d30000.pt', weights_only=False)
# d_model_2 = torch.load('./pth_save/2d30000.pt', weights_only=False)
# d_model_3 = torch.load('./pth_save/3d30000.pt', weights_only=False)


optimizerD = torch.optim.Adam(list(d_model.parameters()) + list(d_model_2.parameters()) + list(d_model_3.parameters()), lr=5e-5, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(g_model.parameters(), lr=1e-5, betas=(0.5, 0.999))
while_number = 0

encoding_texts_batch =np.load(f"./np_data/encoding_texts_batch.npy")
break_batch = np.load(f"./np_data/break_batch.npy")
accompaniment_batch = np.load(f"./np_data/accompaniment_batch.npy")
origin_batch = np.load(f"./np_data/origin_batch.npy")
song_batch = np.load(f"./np_data/song_batch.npy")

while True:
    for epoch in range(0, 87):
        g_model.train()
        d_model.train()
        d_model_2.train()
        d_model_3.train()

        while_number += 1
        string_data=torch.from_numpy(encoding_texts_batch[epoch*5:epoch*5+5]).type(torch.long).to(device)
        breaking_music_data=torch.from_numpy(break_batch[epoch*5:epoch*5+5]).type(torch.float).to(device)
        accompaniment_music_data=torch.from_numpy(accompaniment_batch[epoch*5:epoch*5+5]).type(torch.float).to(device)
        # original_music_data=torch.from_numpy(origin_batch[epoch*5:epoch*5+5]).type(torch.float).to(device)
        song_data=torch.from_numpy(song_batch[epoch*5:epoch*5+5]).type(torch.float).to(device)


        output_real, _ = d_model(song_data)
        output_real_2, _ = d_model_2(accompaniment_music_data)
        output_real_3, _ = d_model_3(torch.logsumexp(torch.stack([accompaniment_music_data, song_data], dim=0), dim=0).detach())

        fake_data, test1, sum_song= g_model(string_data, breaking_music_data) 
        output_fake, output_fake_feat = d_model(fake_data.detach())
        output_fake_2, output_fake_feat2 = d_model_2(test1.detach())
        output_fake_3, output_fake_feat3 = d_model_3(sum_song.detach())

        loss_D_1 = discriminator_total_loss(output_real, output_fake)
        loss_D_2 = discriminator_total_loss(output_real_2, output_fake_2)
        loss_D_3 = discriminator_total_loss(output_real_3, output_fake_3)

        loss_D = loss_D_1 + loss_D_2 + loss_D_3

        optimizerD.zero_grad()
        loss_D.backward()
        optimizerD.step()

        # fake_data, test1, sum_song = g_model(string_data, breaking_music_data) 

        output_real, output_real_feat = d_model(song_data)
        output_real_2, output_real_feat2 = d_model_2(accompaniment_music_data)
        output_real_3, output_real_feat3 = d_model_3(torch.logsumexp(torch.stack([accompaniment_music_data, song_data], dim=0), dim=0).detach())


        output_fake_for_G, output_fake_for_G_feat = d_model(fake_data)
        output_fake_for_G_2, output_fake_for_G_2_feat = d_model_2(test1)
        output_fake_for_G_3, output_fake_for_G_3_feat = d_model_3(sum_song)
        

        loss_l1 = generator_total_loss(
            output_fake_for_G,
            song_data.permute(0,2,1),
            fake_data.permute(0,2,1),
            output_real_feat,
            output_fake_for_G_feat
        )
        loss_l2 = generator_total_loss(
            output_fake_for_G_2,
            accompaniment_music_data.permute(0,2,1),
            test1.permute(0,2,1),
            output_real_feat2,
            output_fake_for_G_2_feat
        )
        loss_l3 = generator_total_loss(
            output_fake_for_G_3,
            (torch.logsumexp(torch.stack([accompaniment_music_data, song_data], dim=0), dim=0).detach()).permute(0,2,1),
            sum_song.permute(0,2,1),
            output_real_feat3,
            output_fake_for_G_3_feat
        )
        
        loss_G =  loss_l1 + loss_l2 + loss_l3
    
        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()
        print(f"step_number : {while_number}, loss_value : {loss_G.item()}")
        #{loss_l1.item()}  
        if while_number % 101 == 1:
            

            with torch.no_grad():
                g_model.eval()
                d_model.eval()
                d_model_2.eval()
                d_model_3.eval()
                
                test_num=epoch % 4
                g_fake_data, test2, sum_song2 = g_model(string_data[test_num:test_num+1, :], breaking_music_data[test_num:test_num+1, :, :])
                #  
                g_fake_data = (g_fake_data[0].permute(1,0).detach().cpu().numpy() / g_fake_data[0].permute(1,0).detach().cpu().numpy().max() * 255).astype(np.uint8)
                img = Image.fromarray(g_fake_data)
                img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                img = img.resize((1800, 70), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label1.config(image=photo)
                img_label1.image = photo  # 가비지 컬렉션 방지


                fake_data = (fake_data[test_num].permute(1,0).detach().cpu().numpy() / fake_data[test_num].permute(1,0).detach().cpu().numpy().max() * 255).astype(np.uint8)
                img = Image.fromarray(fake_data)
                img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                img = img.resize((1800, 70), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label2.config(image=photo)
                img_label2.image = photo 

                song_data_s = (song_data[test_num].permute(1,0).detach().cpu().numpy() / song_data[test_num].permute(1,0).detach().cpu().numpy().max() * 255).astype(np.uint8)
                img = Image.fromarray(song_data_s)
                img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                img = img.resize((1800, 70), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label3.config(image=photo)
                img_label3.image = photo 

                test2 = (test2[0].permute(1,0).detach().cpu().numpy() / test2[0].permute(1,0).detach().cpu().numpy().max() * 255).astype(np.uint8)
                img = Image.fromarray(test2)
                img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                img = img.resize((1800, 70), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label4.config(image=photo)
                img_label4.image = photo  # 가비지 컬렉션 방지

                test1 = (test1[test_num].permute(1,0).detach().cpu().numpy() / test1[test_num].permute(1,0).detach().cpu().numpy().max() * 255).astype(np.uint8)
                img = Image.fromarray(test1 )
                img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                img = img.resize((1800, 70), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label5.config(image=photo)
                img_label5.image = photo  # 가비지 컬렉션 방지


                accompaniment_music_data_s = (accompaniment_music_data[test_num].permute(1,0).detach().cpu().numpy() / accompaniment_music_data[test_num].permute(1,0).detach().cpu().numpy().max() * 255).astype(np.uint8)
                img = Image.fromarray(accompaniment_music_data_s)
                img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                img = img.resize((1800, 70), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label6.config(image=photo)
                img_label6.image = photo 


                breaking_music_data = (breaking_music_data[test_num].permute(1,0).detach().cpu().numpy() / breaking_music_data[test_num].permute(1,0).detach().cpu().numpy().max() * 255).astype(np.uint8)
                img = Image.fromarray(breaking_music_data)
                img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                img = img.resize((1800, 70), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label10.config(image=photo)
                img_label10.image = photo





                sum_song2 = (sum_song2[0].permute(1,0).detach().cpu().numpy() / sum_song2[0].permute(1,0).detach().cpu().numpy().max() * 255).astype(np.uint8)
                img = Image.fromarray(sum_song2)
                img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                img = img.resize((1800, 70), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label7.config(image=photo)
                img_label7.image = photo  # 가비지 컬렉션 방지

                sum_song = (sum_song[test_num].permute(1,0).detach().cpu().numpy() / sum_song[test_num].permute(1,0).detach().cpu().numpy().max() * 255).astype(np.uint8)
                img = Image.fromarray(sum_song)
                img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                img = img.resize((1800, 70), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label8.config(image=photo)
                img_label8.image = photo  # 가비지 컬렉션 방지

                aaa= torch.logsumexp(torch.stack([accompaniment_music_data[test_num], song_data[test_num]], dim=0), dim=0).detach() 
                original_music_data = (aaa.permute(1,0).detach().cpu().numpy() / aaa.permute(1,0).detach().cpu().numpy().max() * 255).astype(np.uint8)
                img = Image.fromarray(original_music_data)
                img = img.transpose(Image.FLIP_TOP_BOTTOM) # 저주파가 아래로 오도록 뒤집기
                img = img.resize((1800, 70), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label9.config(image=photo)
                img_label9.image = photo 

                

                root.update()
            # viewer.imshow(fake_data[0].permute(1,0), original_music_data[0].permute(1,0), g_fake_data[0].permute(1,0))
    
        if while_number % 10000 == 0:
            torch.save(g_model, f"./pth_save/1g{while_number}.pt")
            torch.save(d_model, f"./pth_save/1d{while_number}.pt")
            torch.save(d_model_2, f"./pth_save/2d{while_number}.pt")
            torch.save(d_model_3, f"./pth_save/3d{while_number}.pt")
