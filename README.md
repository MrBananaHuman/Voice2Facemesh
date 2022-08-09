# Voice2Facemesh

음성 신호로부터 [MediaPipe](https://google.github.io/mediapipe/)의 [FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html)를 생성하는 토이 프로젝트입니다.

## 1. Result example
https://user-images.githubusercontent.com/34882690/183585022-a78d6152-2687-4266-a7f3-a825ae051399.mp4

## 2. Dataset    
[HTDF](https://github.com/MRzzm/HDTF)    

## 3. Inference
3.1 init_face 생성  
주어진 얼굴 이미지로부터 3D (x, y, z) FaceMesh 좌표를 생성합니다.   
![image](https://user-images.githubusercontent.com/34882690/183590580-c26797dd-cf0a-4fcf-b66f-d285b360873c.png)     
```python
>>> python3 get_init_face.py
```
3.2 facemesh 생성 & 비디오 생성    
```python
>>> sh test.sh
```
```
* inference options
--checkpoint_path: 모델 path
--init_face_path: txt로 추출한 init_face의 facemesh 좌표
--audio_path: 오디오 파일
--output_path: frame별로 생성된 facemesh 좌표
--text: tts 생성을 위한 입력 (미입력시 오디오 path 활용)

* visualizing options
--init_face_path: txt로 추출한 init_face의 facemesh 좌표
--frame_path: frame별로 생성된 facemesh 좌표
--audio_path: 오디오 파일
--output_image_path: frame별로 생성된 이미지 파일
--output_video_name: 최종 비디오 출력
```
    
## 5. License
[CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
상업적 이용의 경우 bananaband657@gmail.com으로 문의 바랍니다.
