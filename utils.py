import torch
import numpy as np
import librosa
import moviepy.editor as mvp

class Utils():
    def __init__(self):
        super().__init__()
        self.max_frame_num = 200
        self.nMels = 80
        self.fps = 30

    def simple_normalize(self, S):
        return (S - S.min()) / (-S.min())

    def face_normalizing(self, face_arrs):
        max_x = -1
        min_x = 1000
        mul_x = 1

        max_y = -1
        min_y = 1000
        mul_y = 1

        max_z = -1
        min_z = 1000
        mul_z = 1

        modi_face = []
        for face_arr in face_arrs:
            x = face_arr[0]
            y = face_arr[1]
            z = face_arr[2]
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
            if z > max_z:
                max_z = z
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if z < min_z:
                min_z = z
        mul_x = 1 / (max_x - min_x)
        mul_y = 1 / (max_y - min_y)
        mul_z = 1 / (max_z - min_z)

        for face_arr in face_arrs:
            x = (face_arr[0] - min_x) * mul_x
            y = (face_arr[1] - min_y) * mul_y
            z = (face_arr[2] - min_z) * mul_z
            modi_face.append([x, y, z])
        
        return modi_face

    def get_init_face(self, init_face_path):
        data = open(init_face_path, 'r', encoding='utf-8')
        lines = data.readlines()
        init_face = []
        for line in lines:
            line = line.strip()
            x = float(line.split('\t')[0])
            y = float(line.split('\t')[1])
            z = float(line.split('\t')[2])
            init_face.append([x, y, z])
        init_face = self.face_normalizing(init_face)
        return init_face

    def get_wav_embedding(self, wav_file_path):
        my_clip = mvp.AudioFileClip(wav_file_path)
        resampled_audio = my_clip.to_soundarray(fps=16000)[:, 0]
        S = librosa.feature.melspectrogram(resampled_audio, sr=16000, n_mels=self.nMels, hop_length=round(16000/self.fps))
        log_S = librosa.power_to_db(S, ref=np.max)
        norm_S = self.simple_normalize(log_S)

        frame_num = norm_S.shape[1]
        if frame_num > self.max_frame_num:
            norm_S = norm_S[:, :self.max_frame_num]
            frame_num = self.max_frame_num
            print('warning! too long sequences!')

        padding_num = self.max_frame_num - norm_S.shape[1]
        padding_arr = np.zeros((norm_S.shape[0], padding_num))
        padded_voice = np.concatenate((norm_S, padding_arr), axis=1)       
        return frame_num, padded_voice