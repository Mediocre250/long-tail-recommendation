import librosa
import librosa.display
import numpy as np
import os
import subprocess

import matplotlib.pyplot as plt

'''
0.事先把视频里面的音频提取成wav文件
1.输入wav文件，重采样为22050Hz，因为快速傅里叶变换的结果覆盖的是0-22050Hz
2.那个n_fft可以改，但是那个是影响傅里叶变换在频率上的分辨率
3.hop_length可以改，一秒想输出多少n个特征，就把他设成22050//n，最好能整除，比如说我想一秒输出5特征，就像现在这样，那n秒我就会输出
    5 * n + 1个特征，总会多一个。舍掉就好
4.n_mels就是一个特征是一个列向量，你想让这个列向量多长就设置成多大
5.其实还可以设置你想分析的声音频率范围，比如说1000-8000Hz，那就设置成第四种情况，放开注释就可以画图
6.这玩意计算量很大，毕竟是各种积分，如果想加速，可以手动多跑几个，但是别超过四个，IO受不了，硬盘会变成类似随机读写慢的很
'''

def extract_wav(path,csv_path):
    for video in os.listdir(path):
        video_to_WAV_command = ["ffmpeg",
                               '-i',  path+'%06d.mp4' % video,  # input file
                               '-ab', '160k',
                               '-ac', '2',
                               '-ar', '44100',
                               '-vn', csv_path+'%06d.wav' % video
                                ]
        print(video_to_WAV_command)
        subprocess.call(video_to_WAV_command,shell=True)


def extract_feats(csv_path):
    y, sr = librosa.load(path)
    sr1 = 22050
    y1 = librosa.resample(y, sr, sr1)
    # plt.subplot(4,1,1)
    # librosa.display.waveplot(y1, sr1)
    melspec = librosa.feature.melspectrogram(y1, sr1, n_fft=1024, hop_length=4410, n_mels=128)
    logmelspec = librosa.power_to_db(melspec)
    # librosa.display.specshow(logmelspec)
    # plt.subplot(4,1,2)
    # librosa.display.specshow(logmelspec, sr=sr)
    # melspec = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=882, n_mels=128)
    # logmelspec = librosa.power_to_db(melspec)
    # plt.subplot(4, 1, 3)
    # librosa.display.specshow(logmelspec, sr=sr)
    # melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=882, n_mels=128, fmin=1000, fmin=8000)
    # logmelspec = librosa.power_to_db(melspec)
    # plt.subplot(4, 1, 3)
    # librosa.display.specshow(logmelspec, sr=sr)
    # plt.show()
    return logmelspec


if __name__ == "__main__":

    path="/home/..."
    csv_path="/home/..."
    feature_path="/home/..."
    extract_wav(path,csv_path)
    for audio in os.listdir(csv_path):
        feature=extract_feats(csv_path)
        np.save(feature_path + audio, feature)




