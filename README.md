# Voice2Facemesh

음성 신호로부터 [MediaPipe](https://google.github.io/mediapipe/)의 [FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html)를 생성하는 프로젝트입니다.

## Result example
https://user-images.githubusercontent.com/34882690/183585022-a78d6152-2687-4266-a7f3-a825ae051399.mp4

## Dataset
[HTDF](https://github.com/MRzzm/HDTF)

## Inference
1. init_face 생성
![image](https://user-images.githubusercontent.com/34882690/183590580-c26797dd-cf0a-4fcf-b66f-d285b360873c.png)
```python
>>> python3 get_init_face.py
```
2. facemesh 생성 & 비디오 생성
```python
>>> sh test.sh
```

## Training
데이터셋에서 FaceMesh와 음성을 추출하여 학습

## License
[CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
상업적 이용의 경우 bananaband657@gmail.com으로 문의 바랍니다.
