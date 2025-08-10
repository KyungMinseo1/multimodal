import time
import os
from wav_process import load_preprocessor
from wav_process import preprocess_audio_dataset as pad
import cv2
from PIL import Image
from model_inference import load_model, run
import json
import torch
import whisper

# video_path : "temp_file/video"
# audio_path : "temp_file/audio"
# ouput_path : 'temp_file'
# whisper 모델 바꾸기
class StreamingSimulator:
    def __init__(self, video_path: str, audio_path: str, output_path:str, max_duration=None, is_live_recording: bool = False):
        self.video_path = video_path
        self.audio_path = audio_path or video_path
        self.is_live_recording = is_live_recording
        self.max_duration = max_duration

        self.output_preprocess_audio = os.path.join(output_path, 'preprocess_audio')
        self.output_chunks = os.path.join(output_path, 'chunks')
        self.output_json = os.path.join(output_path, 'analysis')
        self.fap = load_preprocessor(self.output_preprocess_audio)
        os.makedirs(self.output_chunks, exist_ok=True)
        os.makedirs(self.output_preprocess_audio, exist_ok=True)
        os.makedirs(self.output_json, exist_ok=True)

        # 모델
        self.face_box, self.e2e_model = load_model()
        self.whisper = whisper.load_model("base")
    
    def start_streaming(self, name):
        """스트리밍 시뮬레이션 시작"""
        print("스트리밍 시작...")
      
        chunk_index = 0
        sleep_count = 0

        while True:
            image_path = f"image_{chunk_index:03d}_{name}.jpg"
            audio_path = f"audio_{chunk_index:03d}_{name}.wav"

            if os.path.exists(os.path.join(self.video_path, image_path)) and os.path.exists(os.path.join(self.audio_path, audio_path)):
                # 새 청크 처리 가능
                self._process_new_chunk(chunk_index, name)
                chunk_index += 1
                sleep_count = 0
            else:
                if sleep_count > 10:
                    break          
                # 잠시 대기 후 다시 확인
                time.sleep(2)
                sleep_count += 1
                continue

    def save_inference_result(self, result_num, result_str, yaw, pitch, noise_num, noise_str, start_time, end_time, stt, json_path):
        new_data = {
            "timestamp": {
                "start": start_time,
                "end": end_time
            },
            "result": {
                "num": result_num,
                "str": result_str
            },
            "pose": {
                "yaw": float(yaw),
                "pitch": float(pitch)
            },
            "noise": {
                "num": noise_num,
                "str": noise_str
            },
            "text": stt
        }

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # 리스트에 새 결과 추가
        data.append(new_data)

        # 다시 저장
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _process_new_chunk(self, chunk_index: int, name:str):
        image_path = f"{self.video_path}/image_{chunk_index:03d}_{name}.jpg"
        audio_path = f"{self.audio_path}/audio_{chunk_index:03d}_{name}.wav"

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore #
        image = Image.fromarray(image_rgb)
        
        pt_path = pad(self.fap, audio_path, f"audio_{int(chunk_index):03d}_{name}.pt")
        audio = torch.load(pt_path).unsqueeze(0)

        if image is not None and audio is not None:
            try:
                (result_num, result_str), (yaw, pitch), (noise_result_num, noise_result_str) = run(self.face_box, self.e2e_model, image, audio)
            except Exception as e:
                print(f"[모델 처리 오류] : {e}")
                result_num, result_str = 0, "모델 처리 실패"
                yaw, pitch = 0.0, 0.0
                noise_result_num, noise_result_str = 0, "잡음 예측 실패"

        result_str = ""
        try:
            result = self.whisper.transcribe(audio_path, language="ko")
            stt = result['text']
            stt = stt.strip() # type: ignore
        except Exception as e:
            print(f"[STT 처리 오류] : {e}")
            stt = ""

        self.save_inference_result(result_num, result_str, yaw, pitch, noise_result_num, noise_result_str, chunk_index*10.0, (chunk_index+1)*10.0, stt, f"{self.output_json}/result.json")

        os.remove(image_path)
        os.remove(audio_path)

    
    def warmup_model(self, jpg_input_shape=(1,3,640,640), jpg2_input_shape=(1,3,448,448), aud_input_shape=(1,25)):
        "빠른 추론을 위한 warmup 기능"
        print("Starting Warmup")
        self.face_box.eval()
        self.e2e_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_jpg_input = torch.rand(jpg_input_shape).to(device) * 255
        dummy_jpg2_input = torch.rand(jpg2_input_shape).to(device) * 255
        dummy_aud_input = torch.randn(aud_input_shape).to(device)
        with torch.no_grad():
            for _ in range(2):  # 2회 정도 실행
                self.face_box(dummy_jpg_input)
                self.e2e_model(dummy_jpg2_input, dummy_aud_input)
        print("Warmup Finished")
    
    