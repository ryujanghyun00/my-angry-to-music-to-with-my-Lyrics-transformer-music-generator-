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
import __main__  
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        self.transformer1s_3 = nn.ModuleList([nn.Transformer(d_model, 8, 1, 1,  batch_first=True) for _ in range(15)])
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

        self.transformer1s_3 = nn.ModuleList([nn.Transformer(d_model, 8, 1, 1,  batch_first=True) for _ in range(15)])
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

        out=x2
        for transformer in self.transformer1s_3:
            out1 = transformer(self.positinoal_encoding(y), self.positinoal_encoding(out))
            out = out1 + out
        mel1 = self.linear_projection1(out.permute(0,2,1)).permute(0,2,1)
        return mel1

class Musiclmsum(nn.Module):
    def __init__(self, number0=None, number1=None):
        super().__init__()

        self.lm0 = Musiclm0().to(device)
        self.lm1 = Musiclm1().to(device)
        if number0 != None and number1 != None:

            __main__.Musiclm0 = Musiclm0
            self.lm0 = Musiclm0().to(device)
            # self.d0 = MultiScaleDiscriminator().to(device)
            self.lm0 = torch.load(
                "./pth_save/3g500000.pt",
                weights_only=False,
            )
            # self.d0 = torch.load(
            #     "./pth_save/3d500000.pt",
            #     weights_only=False,
            # )
            __main__.Musiclm0 = Musiclm1
            self.lm1 = Musiclm1().to(device)
            # self.d1 = MultiScaleDiscriminator().to(device)
            self.lm1 = torch.load(
                "./pth_save/4g325000.pt",
                weights_only=False,
            )
            # self.d1 = torch.load(
            #     "./pth_save/4d325000.pt",
            #     weights_only=False,
            # )
    def forward(self, x, y1):    
        out1 = self.lm0(x, y1)
        out2 = self.lm1(x, out1)
        return out1, out2
class Musiclmsum2(nn.Module):
    def __init__(self, number):
        super().__init__()
        self.ml = Musiclmsum().to(device)
        if number != None:
            self.ml.load_state_dict(torch.load(f"./pth_save/5g100000.pt", map_location=device))
    def forward(self, x, y1):   
        out1, out2= self.ml(x, y1)
        out3 = torch.logaddexp(out1*12, out2*12)/12
        return out1, out2, out3
model0 = Musiclmsum2(None).to(device)
model0.load_state_dict(torch.load(f"./pth_save/6g60000.pt", map_location=device))

model0.to(device)
model0.eval()


with torch.no_grad():
 

    D1_batch = torch.tensor([], dtype=torch.float).to(device)
    encoding_texts = torch.tensor([], dtype=torch.long).to(device)
    
    
    wav, sr = librosa.load(
        './input_data/distorted_song.wav',
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
    

    with open('./input_data/gasa.txt', 'r', encoding='utf-8') as f:
        gasa_text = f.readlines()
        gasa_text = ''.join(gasa_text)
        # print(gasa_text)
        encoding_text, _=gasa_encode(gasa_text)

        encoding_texts = torch.cat((encoding_texts, encoding_text.to(device)), dim=0) 

    encoding_texts = encoding_texts.type(torch.long).to(device)
    D1_batch = D1_batch.permute(0,2,1).type(torch.float).to(device)
    with torch.autocast(
                            "cuda",
                            dtype=torch.bfloat16,
                            enabled=(device == "cuda")
                        ):
        song3_1, song3_2, song3_3  = model0(encoding_texts, D1_batch)
        # song3_3 = torch.logaddexp(song3_1*12, song3_2*12)/12
        
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
        "./output_data/out_song2_1.wav",
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
        "./output_data/out_song2_2.wav",
        wav_recon,
        22050
    )

    


    song4 = song3_3.permute(0,2,1)
            
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
        "./output_data/out_song2_3.wav",
        wav_recon,
        22050
    )

    print("Recon shape:", wav_recon.shape)




with torch.no_grad():
 

    D1_batch = torch.tensor([], dtype=torch.float).to(device)
    encoding_texts = torch.tensor([], dtype=torch.long).to(device)
    
    
    wav, sr = librosa.load(
        './input_data/testtt.mp3',
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
    

    with open('./input_data/test.txt', 'r', encoding='utf-8') as f:
        gasa_text = f.readlines()
        gasa_text = ''.join(gasa_text)
        # print(gasa_text)
        encoding_text, _=gasa_encode(gasa_text)

        encoding_texts = torch.cat((encoding_texts, encoding_text.to(device)), dim=0) 

    encoding_texts = encoding_texts.type(torch.long).to(device)
    D1_batch = D1_batch.permute(0,2,1).type(torch.float).to(device)


    #with torch.autocast("cuda", dtype=torch.bfloat16):
    with torch.autocast(
                                "cuda",
                                dtype=torch.bfloat16,
                                enabled=(device == "cuda")
                            ):
        song3_1, song3_2, song3_3  = model0(encoding_texts, D1_batch)
        # song3_3 = torch.logaddexp(song3_1*12, song3_2*12)/12
        
    
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
        "./output_data/out_song4_1.wav",
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
        "./output_data/out_song4_2.wav",
        wav_recon,
        22050
    )

    


    song4 = song3_3.permute(0,2,1)
            
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
        "./output_data/out_song4_3.wav",
        wav_recon,
        22050
    )

    print("Recon shape:", wav_recon.shape)


with torch.no_grad():
 

    D1_batch = torch.tensor([], dtype=torch.float).to(device)
    encoding_texts = torch.tensor([], dtype=torch.long).to(device)
    
    
    wav, sr = librosa.load(
        './input_data/distorted_song2.mp3',
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
    

    with open('./input_data/gasa2.txt', 'r', encoding='utf-8') as f:
        gasa_text = f.readlines()
        gasa_text = ''.join(gasa_text)
        # print(gasa_text)
        encoding_text, _=gasa_encode(gasa_text)

        encoding_texts = torch.cat((encoding_texts, encoding_text.to(device)), dim=0) 

    encoding_texts = encoding_texts.type(torch.long).to(device)
    D1_batch = D1_batch.permute(0,2,1).type(torch.float).to(device)

    with torch.autocast(
                            "cuda",
                            dtype=torch.bfloat16,
                            enabled=(device == "cuda")
                        ):
        song3_1, song3_2, song3_3  = model0(encoding_texts, D1_batch)
        # song3_3 = torch.logaddexp(song3_1*12, song3_2*12)/12

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
        "./output_data/out_song5_1.wav",
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
        "./output_data/out_song5_2.wav",
        wav_recon,
        22050
    )

    


    song4 = song3_3.permute(0,2,1)
            
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
        "./output_data/out_song5_3.wav",
        wav_recon,
        22050
    )

    print("Recon shape:", wav_recon.shape)
