python3 inference.py \
    --checkpoint_path 'model_output/003189k_state.pt' \
    --init_face_path 'init_face/son.txt' \
    --audio_path 'ttsstream.mp3' \
    --output_path 'inference_frames' \
    --text '안녕하세요. 만나서 반갑습니다.'

python3 visualizing.py \
    --init_face_path 'init_face/son.txt' \
    --frame_path 'inference_frames' \
    --audio_path 'ttsstream.mp3' \
    --output_image_path 'converted_imgs' \
    --output_video_name 'converted_video_3d'
