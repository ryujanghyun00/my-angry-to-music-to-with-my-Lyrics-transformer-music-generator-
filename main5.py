import torch
import torch.nn as nn
import numpy as np
from mylib.gasa_encoding import gasa_encode
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import math 
import librosa
import soundfile as sf
from modules import Postnet, TextPrenet, ConvNorm, Postnet, PositionalEncoding, Prenet
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.enable_flash_sdp(True)
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



model1 = Musiclm2().to(device)

model1 = torch.load(
    "./pth_save/1g820000.pt",
    weights_only=False,
)

model1.to(device)
model1.eval()

with torch.no_grad():
 

    D1_batch = torch.tensor([], dtype=torch.float).to(device)
    encoding_texts = torch.tensor([], dtype=torch.long).to(device)
    
    
    wav, sr = librosa.load(
        './input_data/distorted_song.mp3',
        sr=22050
    )

    mel = librosa.feature.melspectrogram(
                y=wav,
                sr=sr,
                n_fft=1024*2,
                hop_length=1024,
                n_mels=80,
                power=2.0
            )

    # Mel → dB
    mel_db = np.log(mel+1e-5)/12
    D1 = torch.tensor(mel_db).unsqueeze(0).to(device)

    
    if(D1.shape[2]<8000):
        D1 = torch.cat((D1, torch.zeros(1, D1.shape[1], 8000-D1.shape[2]).to(device)), dim=2)
    
    D1_batch = torch.cat((D1_batch, D1), dim=0)  # (Batch, Freq, Time)
    

    with open('./input_data/gasa5.txt', 'r', encoding='utf-8') as f:
        gasa_text = f.readlines()
        gasa_text = ''.join(gasa_text)
        # print(gasa_text)
        encoding_text, _=gasa_encode(gasa_text)

        encoding_texts = torch.cat((encoding_texts, encoding_text.to(device)), dim=0) 

    encoding_texts = encoding_texts.type(torch.long).to(device)
    D1_batch = D1_batch.permute(0,2,1).type(torch.float).to(device)


    with torch.autocast("cuda", dtype=torch.bfloat16):
        song3, song3_1, song3_2 = model1.forward(encoding_texts, D1_batch)
    
    
    song4 = song3.permute(0,2,1)
    
    song4=torch.exp(song4*12)
    mel_db_out = song4.type(torch.float).squeeze(0).cpu().numpy()   
    print('test1')

    wav_recon = librosa.feature.inverse.mel_to_audio(
        mel_db_out,
        sr=22050,
        n_fft=1024*2,
        hop_length=1024,
        n_iter=80
    )

    
    sf.write(
        "./output_data/out_song1.wav",
        wav_recon,
        22050
    )

    song4 = song3_1.permute(0,2,1)
        
    song4=torch.exp(song4*12)
    mel_db_out = song4.type(torch.float).squeeze(0).cpu().numpy()   
    print('test1')

    wav_recon = librosa.feature.inverse.mel_to_audio(
        mel_db_out,
        sr=22050,
        n_fft=1024*2,
        hop_length=1024,
        n_iter=80
    )

    
    sf.write(
        "./output_data/out_song2.wav",
        wav_recon,
        22050
    )

    song4 = song3_2.permute(0,2,1)
        
    song4=torch.exp(song4*12)
    mel_db_out = song4.type(torch.float).squeeze(0).cpu().numpy()   
    print('test1')

    wav_recon = librosa.feature.inverse.mel_to_audio(
        mel_db_out,
        sr=22050,
        n_fft=1024*2,
        hop_length=1024,
        n_iter=80
    )

    
    sf.write(
        "./output_data/out_song3.wav",
        wav_recon,
        22050
    )
    print("Recon shape:", wav_recon.shape)

    