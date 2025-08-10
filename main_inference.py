import subprocess
import time
import os
from typing import Optional
from wav_process import load_preprocessor
from wav_process import preprocess_audio_dataset as pad
import cv2
from PIL import Image
from model_inference import load_model, run
import json
import torch
import whisper


# ouput_path : 'temp_file'
class StreamingSimulator:
    def __init__(self, video_path: str, audio_path: str, output_path:str, max_duration=None, is_live_recording: bool = False, starting_point: float = 0.0, start_idx = 0.0):
        self.video_path = video_path
        self.audio_path = audio_path or video_path
        self.is_live_recording = is_live_recording
        self.max_duration = max_duration
        self.starting_point = starting_point
        self.start_idx = start_idx

        self.output_preprocess_audio = os.path.join(output_path, 'preprocess_audio')
        self.output_chunks = os.path.join(output_path, 'chunks')
        self.output_json = os.path.join(output_path, 'analysis')
        self.fap = load_preprocessor(self.output_preprocess_audio)
        os.makedirs(self.output_chunks, exist_ok=True)
        os.makedirs(self.output_preprocess_audio, exist_ok=True)
        os.makedirs(self.output_json, exist_ok=True)
        
        # 스트리밍 상태
        self.stream_start_time = None
        self.available_duration = 0.0  # 현재 접근 가능한 길이
        self.total_duration = self._get_total_duration()

        # starting_point와 max_duration을 고려한 실제 처리 범위 계산
        if self.max_duration is not None:
            # starting_point부터 max_duration만큼의 범위로 제한
            self.effective_end_time = min(self.total_duration, self.starting_point + self.max_duration)
        else:
            self.effective_end_time = self.total_duration
        
        # 버퍼
        self.chunk_duration = 10.0

        # 모델
        self.face_box, self.e2e_model = load_model()
        self.whisper = whisper.load_model("base")
        
        print(f"초기화 완료 - 총 길이: {self.total_duration:.1f}초")
        if self.max_duration:
            print(f"최대 처리 시간: {self.max_duration:.1f}초로 제한")
        print(f"모드: {'실시간 녹화' if is_live_recording else '완성파일 시뮬레이션'}")
    
    def _get_total_duration(self) -> float:
        """파일의 총 길이 구하기 (완성파일인 경우)"""
        if not os.path.exists(self.video_path):
            return float('inf')
        
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of',
                'default=noprint_wrappers=1:nokey=1', self.video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            print("⚠️ ffprobe로 duration 가져오기 실패, 재시도 대기")
            return 0.0
    
    def _get_current_file_duration(self) -> float:
        """현재 파일의 실제 길이 (실시간 녹화용)"""
        if not os.path.exists(self.video_path):
            return 0.0
        
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of',
                'default=noprint_wrappers=1:nokey=1', self.video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            return duration
        except:
            return 0.0
    
    def _get_available_duration(self) -> float:
        """현재 스트리밍에서 접근 가능한 길이 계산"""
        if self.stream_start_time is None:
            return 0.0
        
        # 스트리밍 시작 후 경과 시간
        stream_elapsed_time = time.time() - self.stream_start_time
        
        if self.is_live_recording:
            # 실제 녹화 중
            actual_recorded = self._get_current_file_duration()
            return min(actual_recorded, stream_elapsed_time)
        else:
            # 완성 파일: starting_point부터 현재까지 접근 가능한 절대 시간
            available_absolute = self.starting_point + stream_elapsed_time
            return min(self.effective_end_time, available_absolute)
    
    def _extract_frame_at_time(self, timestamp: float) -> Optional[bytes]:
        """특정 시점의 프레임을 JPG로 추출"""
        if not os.path.exists(self.video_path):
            return None
            
        try:
            # 올바른 FFmpeg 명령어 순서: -ss는 -i 앞에, -t는 duration
            cmd = [
                'ffmpeg',
                '-ss',
                str(timestamp),
                '-i',
                self.video_path,
                '-vframes',
                '1',
                '-f',
                'image2',
                '-c:v',
                'mjpeg',
                '-y',
                '-'
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"[FFmpeg 프레임 오류] {result.stderr.decode()}")
                return None
            return result.stdout
        except Exception as e:
            print(f"[프레임 추출 예외] {e}")
            return None
    
    def _extract_audio_chunk(self, start_time: float, duration: float) -> Optional[bytes]:
        """특정 구간의 오디오를 WAV로 추출"""
        if not os.path.exists(self.audio_path):
            return None
            
        try:
            # 올바른 FFmpeg 명령어 순서: -ss는 -i 앞에, -t는 duration
            cmd = [
            'ffmpeg', 
            '-ss', str(start_time),
            '-i', self.audio_path,
            '-t', str(duration),
            '-ac', '1',  # 모노로 변환
            '-ar', '48000',  # 48kHz 샘플링 레이트
            '-f', 'wav',
            '-y',  # 덮어쓰기 허용
            '-'
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"[FFmpeg 오디오 오류] {result.stderr.decode()}")
                return None
            return result.stdout
        except Exception as e:
            print(f"[오디오 추출 예외] {e}")
            return None
    
    def start_streaming(self):
        """스트리밍 시뮬레이션 시작"""
        print("스트리밍 시작...")
        self.stream_start_time = time.time()
        self.available_duration = 0.0
        
        chunk_index = 0
        
        while True:
            # 현재 접근 가능한 길이 업데이트
            self.available_duration = self._get_available_duration()
            
            # 새로운 청크를 처리할 수 있는지 확인
            next_chunk_start = self.starting_point + (chunk_index * self.chunk_duration)
            next_chunk_end = next_chunk_start + self.chunk_duration
            
            if self.available_duration >= next_chunk_end:
                # 새 청크 처리 가능
                self._process_new_chunk(chunk_index)
                chunk_index += 1
            else:
                # 아직 데이터가 부족함
                if not self.is_live_recording and self.available_duration >= self.effective_end_time:  # 수정된 부분
                    # 완성파일이고 모든 처리 완료
                    print("스트리밍 완료!")
                    break
                
                # 잠시 대기 후 다시 확인
                time.sleep(0.5)
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

    def _process_new_chunk(self, chunk_index: int):
        start_time = self.starting_point + (chunk_index * self.chunk_duration)  # starting_point 추가
        end_time = start_time + self.chunk_duration
        frame_timestamp = start_time + (self.chunk_duration / 2)

        image = None
        audio = None
        
        print(f"청크 {int(chunk_index+self.start_idx):03d}: {start_time:.1f}~{start_time + self.chunk_duration:.1f}초 처리 중...")

        # 디버깅 출력 추가
        remaining_duration = min(self.chunk_duration, self.available_duration - start_time)
        print(f"Debug: available_duration={self.available_duration:.3f}, start_time={start_time:.3f}, remaining_duration={remaining_duration:.3f}")
        
        # remaining_duration이 음수면 처리하지 않음
        if remaining_duration <= 0:
            print(f"  → 처리 가능한 오디오 길이가 없음 (remaining_duration={remaining_duration:.3f})")
            return
        
        # JPG 프레임 추출 (중간 지점)
        jpg_data = self._extract_frame_at_time(frame_timestamp)
        if jpg_data:
            with open(f"{self.output_chunks}/frame_{int(chunk_index+self.start_idx):03d}.jpg", "wb") as f:
                f.write(jpg_data)
            image = cv2.imread(f"{self.output_chunks}/frame_{int(chunk_index+self.start_idx):03d}.jpg")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
            print(f"  → 프레임 추출 완료 ({frame_timestamp:.1f}초 지점, {len(jpg_data)} bytes)")
        else:
            print(f"  → 프레임 추출 실패")
        
        # WAV 청크 추출 (전체 10초)
        remaining_duration = min(self.chunk_duration, self.available_duration - start_time)
        wav_data = self._extract_audio_chunk(start_time, remaining_duration)
        if wav_data:
            with open(f"{self.output_chunks}/audio_{int(chunk_index+self.start_idx):03d}.wav", "wb") as f:
                f.write(wav_data)
            audio_path = pad(self.fap, f"{self.output_chunks}/audio_{int(chunk_index+self.start_idx):03d}.wav", f"audio_{int(chunk_index+self.start_idx):03d}.pt")
            audio = torch.load(audio_path).unsqueeze(0)
        else:
            print(f"  → 오디오 추출 실패")
        
        print(f"청크 {int(chunk_index+self.start_idx):03d} 처리 완료!")
        print("모델 처리 시작")
        if image is not None and audio is not None:
            try:
                (result_num, result_str), (yaw, pitch), (noise_result_num, noise_result_str) = run(self.face_box, self.e2e_model, image, audio)
            except Exception as e:
                print(f"[모델 처리 오류] : {e}")
                result_num, result_str = 0, "모델 처리 실패"
                yaw, pitch = 0.0, 0.0
                noise_result_num, noise_result_str = 0, "잡음 예측 실패"

        print("모델 처리 완료!")
        print("STT 처리 시작")
        result_str = ""
        try:
            result = self.whisper.transcribe(f"{self.output_chunks}/audio_{int(chunk_index+self.start_idx):03d}.wav", language="ko")
            stt = result['text']
            stt = stt.strip()
        except Exception as e:
            print(f"[STT 처리 오류] : {e}")
            stt = ""
        print("STT 처리 완료!")

        self.save_inference_result(result_num, result_str, yaw, pitch, noise_result_num, noise_result_str, start_time, end_time, stt, f"{self.output_json}/result.json")
        print("청크 처리 완료!\n")
    
    def get_status(self) -> dict:
        """현재 스트리밍 상태 반환"""
        return {
            'available_duration': self.available_duration,
            'total_duration': self.total_duration,
            'is_streaming': self.stream_start_time is not None,
            'progress': self.available_duration / self.total_duration if self.total_duration > 0 else 0
        }
    
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

# 사용 예시
if __name__ == "__main__":
    # 1. 완성된 파일을 실시간 스트리밍처럼 처리
    print("=== 완성파일 시뮬레이션 ===")
    simulator1 = StreamingSimulator("data_inference/02-04-78--1-23-23101300000021-01.mp4", "data_inference/02-04-78--1-23-23101300000021-01.wav", "data_inference", is_live_recording=False, max_duration=120.0, starting_point=1100.0, start_idx=60)
    simulator1.warmup_model()
    simulator1.start_streaming()
    print(f"현재 상태: {simulator1.get_status()}")

    simulator1_1 = StreamingSimulator("data_inference/02-04-78--1-23-23101300000021-01.mp4", "data_inference/02-04-78--1-23-23101300000021-01.wav", "data_inference", is_live_recording=False, max_duration=150.0, starting_point=1220.0, start_idx=72)
    simulator1_1.start_streaming()
    print(f"현재 상태: {simulator1_1.get_status()}")

    simulator2 = StreamingSimulator("data_inference/02-04-78--1-23-23101300000021-02.mp4", "data_inference/02-04-78--1-23-23101300000021-02.wav", "data_inference", is_live_recording=False, max_duration=120.0, starting_point=860.0, start_idx=36)
    simulator2.start_streaming()
    print(f"현재 상태: {simulator2.get_status()}")

    simulator2_1 = StreamingSimulator("data_inference/02-04-78--1-23-23101300000021-02.mp4", "data_inference/02-04-78--1-23-23101300000021-02.wav", "data_inference", is_live_recording=False, max_duration=120.0, starting_point=740.0, start_idx=24)
    simulator2_1.start_streaming()
    print(f"현재 상태: {simulator2_1.get_status()}")

    simulator3 = StreamingSimulator("data_inference/02-04-78--1-23-23101300000021-03.mp4", "data_inference/02-04-78--1-23-23101300000021-03.wav", "data_inference", is_live_recording=False, max_duration=120.0, starting_point=620.0, start_idx=12)
    simulator3.start_streaming()
    print(f"현재 상태: {simulator3.get_status()}")

    simulator4 = StreamingSimulator("data_inference/02-04-78--1-23-23101300000021-04.mp4", "data_inference/02-04-78--1-23-23101300000021-04.wav", "data_inference", is_live_recording=False, max_duration=120.0, starting_point=980.0, start_idx=48)
    simulator4.start_streaming()
    print(f"현재 상태: {simulator4.get_status()}")

    simulator5 = StreamingSimulator("data_inference/02-04-78--1-23-23101300000021-05.mp4", "data_inference/02-04-78--1-23-23101300000021-05.wav", "data_inference", is_live_recording=False, max_duration=120.0, starting_point=500.0, start_idx=0)
    simulator5.start_streaming()
    print(f"현재 상태: {simulator5.get_status()}")
    
    # 2. 실시간 녹화 중인 파일 처리  
    # print("\n=== 실시간 녹화 시뮬레이션 ===")
    # simulator2 = StreamingSimulator("recording.mp4", is_live_recording=True)
    
    