import whisper
from deepface import DeepFace
import tensorflow as tf
import cv2
import os
import sys
import time

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0:All, 1:Filter Info, 2:Filter Warning, 3:Filter Error
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Mematikan peringatan oneDNN
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Gunakan GPU pertama (jika ada). Set ke "" untuk CPU saja.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print(tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Is GPU available?", tf.test.is_gpu_available())
print("Device Name:", tf.test.gpu_device_name())



def transcript(video_path):
   # 1. Transkripsi Audio (Pilihan Kata)
   print("--- Memulai Transkripsi Whisper ---")
   model_whisper = whisper.load_model("medium") # Pilihan model: tiny, base, small, medium, large
   transcription = model_whisper.transcribe(video_path, language="id")
   return transcription


def transcript(video_path):
   # 1. Transkripsi Audio (Pilihan Kata)
   print("--- Memulai Transkripsi Whisper ---")
   model_whisper = whisper.load_model("medium") # Pilihan model: tiny, base, small, medium, large
   transcription = model_whisper.transcribe(video_path, language="id")
   return transcription


def process_video_with_overlay(input_path, output_path):
   # 1. Buka Video Source
   cap = cv2.VideoCapture(input_path)
   if not cap.isOpened():
      print("Error: Video tidak ditemukan.")
      return

   # Ambil properti video asli
   width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps         = cap.get(cv2.CAP_PROP_FPS)
   total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Ambil total frame
   fourcc      = cv2.VideoWriter_fourcc(*'mp4v') 
   current_frame = 0
   
   
   start_time = time.time()
   
   # 2. Siapkan Video Writer
   out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

   dominant_emotion = "Mencari..."
   rect_coords = None

   while cap.isOpened():
      ret, frame = cap.read()
      if not ret: break
      
      time_string = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

      current_frame += 1

      # Update analisis setiap 3 frame untuk menghemat GPU RTX 4050 Anda
      if current_frame % 3 == 0:
         try:
            progress_msg = f"Progress: {current_frame}/{total_frames} frame ({(current_frame/total_frames)*100:.2f}%) -- {time_string}"
                        
            print(progress_msg)
            
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            
            dominant_emotion = analysis[0]['dominant_emotion']
            rect_coords = analysis[0]['region']
         except Exception as exception:
            print(f"Analisis gagal pada frame ini, melanjutkan ke frame berikutnya... Error: {exception}")
            pass

      # Gambar ulang hasil terakhir pada SETIAP frame agar tidak kelap-kelip
      if rect_coords:
         x, y, w, h = rect_coords['x'], rect_coords['y'], rect_coords['w'], rect_coords['h']
         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
         cv2.putText(frame, f"Emosi: {dominant_emotion.upper()}", (x, y - 10), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

      out.write(frame)

   # 5. Selesai
   cap.release()
   out.release()
   print(f"\nSelesai! Video disimpan di: {output_path}")


# for a  realtime analysis using camera
def start_realtime_emotion():
   cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # use CAP_V4L2 for 

   cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
   cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

   if not cap.isOpened():
      print("Error: Kamera tidak terdeteksi. Jika di WSL2, pastikan usbipd sudah terhubung.")
      return

   print("Tekan 'q' untuk berhenti...")

   while True:
      ret, frame = cap.read()
      if not ret:
         break

      try:
         # 2. Analisis Wajah (Hanya emosi agar FPS tetap tinggi)
         # detector_backend 'opencv' adalah yang tercepat untuk real-time
         results = DeepFace.analyze(frame, 
                                    actions=['emotion'], 
                                    enforce_detection=False, 
                                    detector_backend='opencv')

         # 3. Gambar hasil analisis ke layar
         for face in results:
               x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
               emotion = face['dominant_emotion']

               # Gambar kotak di wajah
               cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
               
               # Tulis teks emosi
               cv2.putText(frame, f"Emosi: {emotion.upper()}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

      except Exception:
         # Lewati jika wajah tidak terdeteksi dengan jelas di frame tersebut
         pass

      # 4. Tampilkan jendela video
      cv2.imshow('AIView Real-time Analysis', frame)

      # Berhenti jika menekan tombol 'q'
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break

   # Bersihkan resource
   cap.release()
   cv2.destroyAllWindows()
   
# DATASET folder location
dataset_folder = "/mnt/c/DATA D/aa-kuliah/skripsi/DATASET"





# video_file = "videos/tips_interview_1080p.mp4" # Ganti dengan nama file video Anda

# # Transcript
# transcription = transcript(video_file)
# print("\n--- Transkripsi ---")
# print(transcription)






# """
#    For a real-time analysis using camera, you can call the function start_realtime_emotion() at the end of this script.
# """

# video_file = "videos/tips_interview_1080p.mp4" # Ganti dengan nama file video Anda

# if os.path.exists(video_file):
#    # If you want to save the video with emotion overlay, uncomment the lines below and specify the output path.
#    # output_video = "/mnt/c/Users/zero/Documents/result.mp4"
#    output_video = "/mnt/c/DATA D/aa-kuliah/skripsi/AIVIEW/output/output2.mp4"
#    process_video_with_overlay(video_file, output_video)
   
# else:
#    print(f"Error: File '{video_file}' tidak ditemukan.")

# # start_realtime_emotion()




limit = 1 # Batasi jumlah video yang diproses untuk demo
count = 0

for root, dirs, files in os.walk(f"{dataset_folder}/first-impressions/train/"):
   for file in files:
      if file.endswith(".mp4"):
         video_path = os.path.join(root, file)
         print(f"Processing: {video_path}")
         output_video = video_path.replace("train", "output").replace(".mp4", "_processed.mp4")

         if not os.path.exists(root.replace("train", "output")):
            os.makedirs(root.replace("train", "output"))
            
         process_video_with_overlay(video_path, output_video)
         count += 1
         if count >= limit:
            break
   if count >= limit:
      break












