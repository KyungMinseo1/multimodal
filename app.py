import os, json, cv2, pyaudio, wave, threading, time
from flask import Flask, request, render_template, session, jsonify
from main_inference import StreamingSimulator
from main_llm import GenerateFeedback

# Flask 앱 초기화
app = Flask(__name__)
app.secret_key = "654951256549874"  # 실제 시크릿 키를 설정하세요

simulator = StreamingSimulator('temp_file/video', 'temp_file/audio', 'temp_file', is_live_recording=True)
simulator.warmup_model()
feedback_generator = GenerateFeedback()

recording_active_v2 = False
segment_counter = 0

@app.route("/", methods=["GET", "POST"])
def home():
    """홈페이지 라우트"""
    if request.method == "POST":
        name = request.form.get("name")
        topic = request.form.get("topic")
        session["name"] = name
        session["topic"] = topic
        return render_template("feedback.html")
    return render_template("home.html")

@app.route("recording", methods=["POST"])
def recording():
    global recording_active_v2, segment_counter
    
    if recording_active_v2:
        return jsonify({"status": "already_recording"})
    
    recording_active_v2 = True
    segment_counter = 0
    
    def recording_segments():
        global recording_active_v2, segment_counter
        
        while recording_active_v2:
            segment_counter += 1
            name = session['name']
            
            # 웹캠 설정 (이미지 캡쳐용)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # 오디오 설정
            p = pyaudio.PyAudio()
            audio_format = pyaudio.paInt16
            channels = 1
            rate = 48000
            chunk = 1024
            
            audio_stream = p.open(format=audio_format,
                                 channels=channels,
                                 rate=rate,
                                 input=True,
                                 frames_per_buffer=chunk)
            
            audio_frames = []
            image_captured = False
            
            # 10초 동안 오디오 녹음 + 1번 이미지 캡쳐
            start_time = time.time()
            while time.time() - start_time < 10 and recording_active_v2:
                # 첫 번째 프레임에서 이미지 캡쳐 (10초마다 1장)
                if not image_captured:
                    ret, frame = cap.read()
                    if ret:
                        image_filename = f"temp_file/images/image_{segment_counter:03d}_{name}.jpg"
                        cv2.imwrite(image_filename, frame)
                        image_captured = True
                
                # 오디오 데이터 수집
                try:
                    data = audio_stream.read(chunk, exception_on_overflow=False)
                    audio_frames.append(data)
                except:
                    pass
                
                time.sleep(0.1)  # CPU 사용량 줄이기
            
            # 리소스 해제
            cap.release()
            audio_stream.stop_stream()
            audio_stream.close()
            p.terminate()
            
            # 오디오 파일 저장
            if audio_frames:
                audio_filename = f"temp_file/audio/audio_{segment_counter:03d}_{name}.wav"
                wf = wave.open(audio_filename, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(48000)
                wf.writeframes(b''.join(audio_frames))
                wf.close()
    
    # 녹화 스레드 시작
    recording_thread = threading.Thread(target=recording_segments)
    recording_thread.start()
    
    return jsonify({"status": "recording_started", "version": "v2", "segment_duration": "10_seconds", "capture_mode": "image"})

@app.route("/stop_recording_v2", methods=["POST"])
def stop_recording_v2():
    global recording_active_v2
    
    if not recording_active_v2:
        return jsonify({"status": "not_recording"})
    
    recording_active_v2 = False
    
    return jsonify({"status": "recording_stopped", "version": "v2", "total_segments": segment_counter})


@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    feedback = None
    if request.method == "POST":
        name = session.get("name", "사용자")
        topic = session.get("topic", "기본 주제")
        result_path = f'result_{name}_modified.json'
        data = os.path.join('analysis', result_path)
        
        try:
            with open(data, 'r', encoding='utf-8') as f:
                result = json.load(f)
            feedback = feedback_generator.generate(topic, name, result)
        except FileNotFoundError:
            feedback = f"'{name}' 사용자의 데이터 파일을 찾을 수 없습니다."
        except Exception as e:
            feedback = f"오류가 발생했습니다: {str(e)}"
    
    return render_template("feedback.html", feedback=feedback)

if __name__ == "__main__":
    app.run(debug=True)