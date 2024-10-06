import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2

sr = 16000

def audio_to_mel(audio_file, sr=16000, n_fft=1024, hop_length=256, n_mels=256):
    # 加载音频文件
    y, sr = librosa.load(audio_file, sr=sr)
    # 计算梅尔频谱
    # y=y[:int(16000*4.08)]
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,win_length=1024)

    # 转换为对数刻度
    log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)

    return log_mel_spec
# #
# # # 指定输入的音频文件路径
root='/Users/cxz/PycharmProjects/UMSS-demo/res'
for exp in os.listdir(root):
    for m in os.listdir(f'/{root}/{exp}'):
        for s in os.listdir(f'/{root}/{exp}/{m}'):
            for audio_file in os.listdir(f'/{root}/{exp}/{m}/{s}'):
                if audio_file.endswith('.wav'):
                    audio_file = f'/{root}/{exp}/{m}/{s}/{audio_file}'
                    # 将音频文件转换为 Mel 频谱
                    mel_spec_uint8 = audio_to_mel(audio_file)/80*255
                    # 将 Mel 频谱转换为图像数据类型 uint8，并进行归一化到 [0, 255]
                    # print(mel_spec,np.max(mel_spec),np.min(mel_spec))
                    # mel_spec_uint8 = librosa.util.normalize(mel_spec) * 255
                    mel_spec_uint8 = mel_spec_uint8.astype(np.uint8)
                    mag_color = cv2.applyColorMap(mel_spec_uint8, cv2.COLORMAP_INFERNO)[::-1, :, :]
                    print(mag_color.shape)
                    # 将 Mel 频谱保存为图像文件
                    output_file = audio_file.replace('.wav','_mel.png')
                    cv2.imwrite(output_file, mag_color)