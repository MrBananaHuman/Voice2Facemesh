import torch
from transformer import TransformerModel
import tqdm
import os, glob
import argparse
import utils
from gtts import gTTS

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument('--init_face_path', type=str, 
					help='Filepath of txt that contains facemesh to use', required=True)     
parser.add_argument('--audio_path', type=str, 
					help='Filepath of audio file to use as raw audio source', required=True)
parser.add_argument('--output_path', type=str, help='Output path to save coordinate of facemesh', required=True)
parser.add_argument('--text', type=str, help='Script for TTS.', required=False)

args = parser.parse_args()

utils = utils.Utils()

def predict(model, embedded_audio, init_face, audio_frame_num):
    embedded_audio = torch.tensor(embedded_audio, dtype=torch.float32).unsqueeze(0)
    embedded_audio = embedded_audio.permute(2, 0, 1).to(device) 

    x, y, z = model(init_face, embedded_audio)  

    files = glob.glob(args.output_path + '/*')
    for f in files:
        os.remove(f)

    for i in tqdm.tqdm(range(min(200, audio_frame_num+5))):
        output = open(args.output_path + '/frame%03d.txt' % i, 'w', encoding='utf-8')
        for j, x_sample in enumerate(x[i, 0, :]):
            output.write(str(x_sample.item()) + '\t' + str(y[i, 0, j].item()) + '\t' + str(z[i, 0, j].item()) + '\n')
        output.close()

def main():
    model_path = args.checkpoint_path
    model = TransformerModel(max_len=200)
    if device == 'cpu':
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        model.to(device)
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)
    
    if args.text is not None:
        tts = gTTS(args.text, lang='ko')
        tts.save('ttsstream.mp3')
    audio_frame_num, embedded_audio = utils.get_wav_embedding(args.audio_path)
    init_face = utils.get_init_face(args.init_face_path)
    init_face = torch.tensor(init_face).unsqueeze(0).unsqueeze(0).to(device)
    predict(model, embedded_audio, init_face, audio_frame_num)

if __name__ == '__main__':
    main()
 