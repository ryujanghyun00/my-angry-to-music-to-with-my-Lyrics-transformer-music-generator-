import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import numpy as np
import torch.optim as optim
import os
from mylib.gasa_encoding import gasa_encode
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

model = torch.load('./pth_save/1g100000.pt', weights_only=False)

model.eval()
with torch.no_grad():
    
    SAMPLE_RATE = 22050
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    m_min =-21
    m_max = 13

    mel_transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    inverse_mel = T.InverseMelScale(n_stft=N_FFT // 2 + 1, n_mels=N_MELS, sample_rate=SAMPLE_RATE)
    griffin_lim = T.GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH)


    D1_batch = torch.tensor([], dtype=torch.float)
    encoding_texts = torch.tensor([], dtype=torch.long)
    
    waveform1, sr1 = torchaudio.load('./input_data/song.mp3')
    if waveform1.shape[0] > 1:
        waveform1 = torch.mean(waveform1, dim=0, keepdim=True)
    if sr1 != SAMPLE_RATE:
        waveform1 = T.Resample(sr1, SAMPLE_RATE)(waveform1)
    mel = mel_transform(waveform1)
    D1 = torch.log(mel + 1e-9)
    
    # m_min, m_max = log_mel.min(), log_mel.max()

 
    D1 = torch.cat((torch.ones(1, D1.shape[1], 1), D1), dim=2)

    if(D1.shape[2]<16000):
        D1 = torch.cat((D1, torch.zeros(1, D1.shape[1], 16000-D1.shape[2])), dim=2)
    
    D1_batch = torch.cat((D1_batch, D1), dim=0)  # (Batch, Freq, Time)
    

    with open('./input_data/gasa.txt', 'r', encoding='utf-8') as f:
        gasa_text = f.readlines()
        gasa_text = ''.join(gasa_text)
        # print(gasa_text)
        encoding_text=gasa_encode(gasa_text)

        encoding_texts = torch.cat((encoding_texts, encoding_text), dim=0) 

    print(encoding_texts)
    encoding_texts = encoding_texts.type(torch.long).to(device)
    D1_batch = D1_batch.permute(0,2,1).type(torch.float).to(device)
    embeddings1, embedding2, song = model(encoding_texts, D1_batch)
    embeddings1 = embeddings1.permute(0,2,1).cpu()
    embedding2 = embedding2.permute(0,2,1).cpu()
    song = song.permute(0,2,1).cpu()
    recon_mel = torch.exp(embeddings1)
    
    # 복원 (Griffin-Lim)
    spec = inverse_mel(recon_mel)
    recon_waveform = griffin_lim(spec)

    # [SAVE] 파일 저장
    torchaudio.save("./output_data/out_song1.wav", recon_waveform, SAMPLE_RATE)

    # denorm_mel = ((embedding2 + 1) / 2) * (m_max - m_min) + m_min

    recon_mel = torch.exp(embedding2)
    
    # 복원 (Griffin-Lim)
    spec = inverse_mel(recon_mel.to("cpu"))
    recon_waveform = griffin_lim(spec)

    # [SAVE] 파일 저장
    torchaudio.save("./output_data/out_song2.wav", recon_waveform, SAMPLE_RATE)
    

    recon_mel = torch.exp(song)
    
    # 복원 (Griffin-Lim)
    spec = inverse_mel(recon_mel.to("cpu"))
    recon_waveform = griffin_lim(spec)

    # [SAVE] 파일 저장
    torchaudio.save("./output_data/out_song3.wav", recon_waveform, SAMPLE_RATE)
   
    print(f"저장 완료: ./output_data/out_song.wav")
    print(f"출력 형태: {recon_waveform.shape}")