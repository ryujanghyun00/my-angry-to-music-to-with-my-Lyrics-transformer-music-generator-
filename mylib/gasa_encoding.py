from jamo import h2j, j2hcj
import torch

def gasa_encode(text):
    
    charset = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄶㄳㄵㄺㄼㄽㄾㄿㅀㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣabcdefghijklmnopqrstuvwxyz1234567890`'\"\\?/><.,!()~@#$%^&*-_+= \n\r"
    torch_text=torch.tensor([0])
  
    jamo_text = j2hcj(h2j(text))
    jamo_text = jamo_text.lower()
    print('자모수 확인 ' + str(jamo_text.__len__()))
    for char in jamo_text:
        # print(char)
        if char in charset:
            index = charset.index(char)
            # print(index)
            list_input = index+2
            torch_text = torch.cat((torch_text, torch.tensor([list_input])), dim=0)

    if(torch_text.shape[0]<4000):
        torch_zeros_input = torch.ones((4000 - torch_text.shape[0]))
        torch_text = torch.cat((torch_text, torch_zeros_input), dim=0)
    torch_text = torch_text.unsqueeze(0)
    return torch_text

if __name__ == "__main__":
    print(gasa_encode("안녕하세요 ACB"))

