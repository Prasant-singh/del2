from flask import Flask, jsonify ,request
from flask_cors import CORS
from multiprocessing import Process,Event,Queue
from threading import Thread
from src.face_registration import face_detect
from threading import Thread
from src.get_embedding import get_embedding
from scipy.spatial.distance import cosine
import os, shutil, pickle,random,string,glob
from collections import Counter
from datetime import datetime
from main import process_camera_stream,frame_reader
import numpy as np


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def apply_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, DELETE, PUT"
    response.headers["Access-Control-Max-Age"] = "3600"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, SagAuthToken, Access-Control-Allow-Headers, X-Requested-With, remember-me, Authorization"
    return response


UPLOAD_FOLDER = 'Data Uploaded'
VALIDATION_FACE = 'face_validation'
ALLOWED_EXTENSIONS = {'mp4','webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VALIDATION_FACE'] = VALIDATION_FACE
face_embedding_file_path = r'src\embedding_dir\face_embeddings.pkl'
if not os.path.exists(app.config['UPLOAD_FOLDER']):os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok=True)
if not os.path.exists(app.config['VALIDATION_FACE']):os.makedirs(app.config['VALIDATION_FACE'],exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/upload-data',methods=['POST'])
def upload():
    print("data uploading started...........")
    print(request.content_type)
    name = str(request.form['name'])
    empid = str(request.form['empid'])
    video_file = request.files['video']
    person_folder = os.path.join(app.config['UPLOAD_FOLDER'],f"{name}_{empid}")
    if not os.path.exists(person_folder):os.makedirs(person_folder,exist_ok=True)
    filename = f"{name}_{empid}.mp4"
    filepath = os.path.join(person_folder, filename)


    if video_file.filename == '':
        return jsonify({'status': 'No selected file'}), 400
    if 'video' in request.files:
        try:
            file = request.files['video']
            with open(filepath,'wb') as f:
                f.write(file.read())
                # Background function to run both steps
                def background_job():
                    print(f"[{name}] --> Starting face detection...")
                    face_detect(name, empid, person_folder, filepath, 220)
                    print(f"[{name}] --> Face detection completed. Starting embedding...")
                    get_embedding(dataset_dir="Data Uploaded", device='cpu', validation_embedding=False)   # dataset_dir, device,validation_embedding
                    print(f"[{name}] --> Embedding completed.")

                # Run background job in a separate thread
                bg_thread = Thread(target=background_job)
                bg_thread.start()


                print(f"Genearting Embedding for : {name}")
                return jsonify({'status': 'successfully received and saved your data', 'filename': filename}), 200
        except Exception as e:
            return jsonify({'status': 'failed to receive valid data'}), 400
        



@app.route('/validate-face', methods=['POST'])
def validate_face():
    try:
        print("Face Validating ......")
        name = str(request.form['name'])
        empid = str(request.form['empid'])
        video_file = request.files.get('video')
        validation_folder = app.config['VALIDATION_FACE']
        validation_person_folder = os.path.join(validation_folder, f"{name}_{empid}")


        filename = f"{name}_{empid}.mp4"
        filepath = os.path.join(validation_person_folder, filename)


        if not video_file or video_file.filename == '':
            return jsonify({'status': 'No selected file'}), 400

        # Save uploaded video
        with open(filepath, 'wb') as f:
            f.write(video_file.read())

        
        T1 = Thread(target=face_detect, args=[name, empid, validation_person_folder, filepath, 7])
        T1.start()
        T1.join()
        new_user_embs = get_embedding( dataset_dir=validation_folder, device='cpu', validation_embedding=True)

        with open(face_embedding_file_path, 'rb') as file:
            registered_user_embs = pickle.load(file)

        best_score = float("inf")
        best_match = "unidentified"
        final_output_ls = []
        if len(new_user_embs)>1:
            for new_user, embeddings in new_user_embs.items():
                for new_embedding in embeddings:
                    # match_dict = {}
                    for registered_user, registered_embeds in registered_user_embs.items():
                        for registered_embed in registered_embeds:
                            # print(f"{new_user} ======> {registered_user}")
                            distance = cosine(new_embedding.flatten(), registered_embed.flatten())
                            if distance < best_score:
                                best_score = distance
                                best_match = registered_user
                    # best_match if best_score < 0.05 else "Unknown"  # 95% similarity 
                    final_output_ls.append({"Emp": best_match if best_score < 0.09 else "Unknown", "Dist": best_score})


        emp_counter = Counter(entry['Emp'] for entry in final_output_ls)
        if not emp_counter:
            final_output = "Unknown"
            is_registered = None
        else:
            final_output = emp_counter.most_common(1)[0][0]
            is_registered = final_output != "Unknown"

        # final_output = emp_counter.most_common(1)[0][0]
        # is_registered = False if final_output == "Unknown" else True
        print(f"Person name {name} Matched with --->  {final_output}")
        try:
            if os.path.exists(validation_folder):
                shutil.rmtree(validation_folder)
        except Exception as delete_err:
            print("folder is not deleted.")
        print("output is sent....")
        return jsonify({
            'EmpName': name,
            "is_registered":is_registered,
            "name":final_output}), 200

    except Exception as e:
        print("Exception occurred:", e)
        return jsonify({'status': 'failed to receive valid data', 'error': str(e)}), 400





@app.route('/process_stream', methods=['POST'])
def process_stream():
    data = request.get_json()
    urls = data.get("urls")   #
    processes = []
    all_queues_for_monitoring = []

    if not urls and not isinstance(urls, list):
        return jsonify({
            "status": "error plz send a urls in a list e.g [url1,url2....,urln]",
        }), 400
    
    if urls and isinstance(urls, list):
        for  idx,url in enumerate(urls):  
            q = Queue(maxsize=500) 
            e = Event()
            all_queues_for_monitoring.append(q) 


            filename = os.path.basename(url)
            name, ext = os.path.splitext(filename)
            dt,tm,cmid = name.rsplit("_")
            final_dt_tmstmp = datetime.strftime(datetime.strptime(" ".join([dt,tm]), "%d-%m-%Y %H-%M-%S"), "%d/%m/%Y %H:%M:%S")


            producer_thread = Thread(
                target=frame_reader,
                args=(url, cmid, q, e),
                name=f"FrameReader-Cam{cmid}",
                daemon=True
            )
            producer_thread.start()


        ## frame queue processing
            p = Process(target=process_camera_stream, args=(all_queues_for_monitoring[idx],cmid,final_dt_tmstmp,e))   
            p.start()

            processes.append({
                "pid": p.pid,
                "cam_id": cmid,
                "timestamp": final_dt_tmstmp,
                "video": url
            })

        return jsonify({
            "status": "Processing started",
            "details": processes
        }), 200




if __name__ == "__main__":
    app.run(host="0.0.0.0",port=2565,debug=True)
