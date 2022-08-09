import os
import matplotlib.pyplot as plt
import glob
import tqdm
import moviepy.editor as mvp
import utils
import argparse

parser = argparse.ArgumentParser()


parser = argparse.ArgumentParser()
parser.add_argument('--init_face_path', type=str, 
					help='Filepath of txt that contains facemesh to use', required=True)
parser.add_argument('--frame_path', type=str, 
					help='Filepath of facemesh txt files for each frame', required=False)                 
parser.add_argument('--audio_path', type=str, 
					help='Filepath of audio file', required=False)
parser.add_argument('--output_image_path', type=str, help='Output path to save converted frame images', required=True)
parser.add_argument('--output_video_name', type=str, help='Output path to save video', required=False)

args = parser.parse_args()

utils = utils.Utils()

eye_points = [
        33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,
        263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249,
        468,
        473,
        70,63,105,66,107,55,65,52,53,46,
        300,293,334,296,336,285,295,282,283,276,
        78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95,
        61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,
        10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109
    ]

def main():
    init_face = utils.get_init_face(args.init_face_path)

    total_frames = []
    for x in os.walk(args.frame_path):
        for y in glob.glob(os.path.join(x[0], '*.txt')):
            total_frames.append(y)

    images_coor = {}
    for frame in total_frames:
        data = open(frame, 'r', encoding='utf-8')
        lines = data.readlines()
        x_coor = []
        y_coor = []
        z_coor = []
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) > 0:
                additional_x = float(line.split('\t')[0]) / 2
                additional_y = float(line.split('\t')[1]) / 2
                additional_z = float(line.split('\t')[2]) / 2

                if i < 468 or i == 468 or i == 473:
                    x = init_face[i][0] - additional_x
                    y = init_face[i][1] - additional_y
                    z = init_face[i][2] - additional_z
                    x_coor.append(x)
                    y_coor.append(y)
                    z_coor.append(z)         
        images_coor[frame.split('/')[-1].split('.')[0]] = {'x':x_coor, 'y':y_coor, 'z':z_coor}

    sorted_dict = sorted(images_coor.items())

    output_dir = args.output_image_path
    img_array = []

    files = glob.glob(output_dir + '/*')
    for f in files:
        os.remove(f)

    plt.rcParams['grid.color'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes3d.grid'] = False

    for frame in tqdm.tqdm(sorted_dict):
        frame_name = frame[0]
        x_coor = frame[1]['x']
        y_coor = frame[1]['y']
        z_coor = frame[1]['z']
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.set(facecolor='black')   # 'w' 
        ax.scatter(x_coor, y_coor, z_coor, color='white', s=50)
        ax.view_init(90, 90)
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')

        ax.set_xlim(0,1)
        ax.set_ylim(0,1)

        output_file_name = output_dir + '/' + frame_name + '.png'
        fig.savefig(output_file_name)
        plt.close()
        img_array.append(mvp.ImageClip(output_file_name).set_duration(1/30))

    if args.output_video_name is not None:
        video = mvp.concatenate(img_array, method="compose")
        video.write_videofile(args.output_video_name + ".mp4", fps=30)
    
    if args.audio_path is not None:
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio_path, args.output_video_name + ".mp4", args.output_video_name + "_with_audio.mp4")
        print(command)
        os.system(command)

if __name__ == '__main__':
    main()

    



