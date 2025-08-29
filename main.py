from multiprocessing import Process,Event,Queue
import cv2 as cv
import time
from src.util import idintifie_person, log_face_info
from collections import defaultdict
from datetime import timedelta,datetime
import pandas as pd 
from collections import Counter
import copy
import numpy as np 
import threading
import os ,shutil,queue,time ,av 
start = time.time()



def frame_reader(video_path, cam_id, frame_queue, stop_event, starting_time=None):
    """
    Frame reader using PyAV for decoding (instead of cv2.VideoCapture),
    but keeping retry, queue, frame skipping, and encoding logic from the first function.
    """
    print(f"Thread started for it ------->  {video_path} ✅✅✅")

    frame_count = 0
    frame_interval = 11
    max_retries = 3
    retry_delay = 0.2  # seconds
    processed_frame_count = 1

    def open_container():
        """Helper to open RTSP/video with retries using PyAV"""
        for attempt in range(max_retries):
            try:
                container = av.open(
                    video_path,
                    options={
                        "rtsp_transport": "tcp",   # udp, tcp, udp_multicast, http
                        "stimeout": "3000000",  # 3s timeout
                        "fflags": "nobuffer",
                        "flags": "low_delay",
                        "strict": "experimental",
                        # "reorder_queue_size": "0",   # Disable frame reordering
                        "err_detect": "ignore_err"
                    }
                )
                stream = container.streams.video[0]
                stream.thread_type = "AUTO" 
                return container,stream,True if starting_time is None else False 
            except Exception as e:
                print(f"[av {cam_id}] Retry {attempt+1}/{max_retries} failed: {e}")
                time.sleep(retry_delay)
        return None

    # --- Open container ---
    container , stream , live = open_container()
    fps = float(stream.average_rate)  #frame_time_base = 1.0 / fps if fps > 0 else 0.033
    # print(f"camid {cam_id} fps   :  {fps}")
    if container is None:
        print(f"❌ [av {cam_id}] Failed to open video stream: {video_path}")
        frame_queue.put(None)
        return

    try:
        for frame in container.decode(stream):  #video=0
            if stop_event.is_set():
                break

            frame_count += 1
            if frame_count % frame_interval == 0:
                try:
                    img = frame.to_ndarray(format="bgr24")

                    # Timestamp (PyAV gives PTS, but we can fallback to current time if missing)
                    # if starting_time is None:
                    #     current_msec = (
                    #         frame.pts * frame.time_base * 1000
                    #         if frame.pts is not None and frame.time_base is not None
                    #         else time.time() * 1000
                    #     )
                    # else:
                    current_msec = None
                    if live:
                        now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        frame_time = datetime.strptime(now_str, "%d/%m/%Y %H:%M:%S")
                        current_msec = frame_time
                    else:
                        current_msec = (frame_count / fps) * 1000 

                    # Encode as JPEG or png
                    ret, encoded_frame = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 95])
                    # ret, encoded_frame = cv.imencode(".png", img, [cv.IMWRITE_PNG_COMPRESSION, 3])

                    if ret:
                        frame_queue.put((encoded_frame.tobytes(), current_msec, frame_count))
                        print(f"CamID : {cam_id} -------------------------------------------------->  {processed_frame_count}")
                        processed_frame_count += 1 
                except queue.Full:
                    print(f"Queue is Full for it : {video_path}")

    except Exception as e:
        print(f"[av {cam_id}] Error during decoding: {e}")

    finally:
        container.close()
        frame_queue.put(None)
        print(f"Thread Finished for it ------->  {video_path} ")

def process_camera_stream(queue_frames, cam_id, starting_time=None, stop_event=None):
    def final_names(sample, IsCrowd, min_score=82, min_freq=2):
        """
        Returns a list of finalized detected person details:
        (face, date, time, sim_score, status, name, movement)
        """
        if IsCrowd:
            result = []
            all_detections = [
                item for frame in sample
                for items in frame.values()
                for item in items
            ]

            name_freq = Counter(item['NAME'] for item in all_detections)

            for item in all_detections:
                name = item['NAME']
                score = item['SIM_SCORE']
                if name in ("Unidentified", "No face detected"):
                    continue
                if name_freq[name] >= min_freq or (name_freq[name] == 1 and score >= min_score):
                    result.append((
                        item['ID'],
                        item['FACE'],
                        item['DATE'],
                        item['TIME'],
                        item['SIM_SCORE'],
                        item['STATUS'],
                        item['NAME'],
                        item['movement']
                    ))
            return result

        else:
            detections_by_id = defaultdict(list)
            for frame in sample:
                for camid, boxes in frame.items():
                    for box in boxes:
                        detections_by_id[box['ID']].append(box)

            final_results = []

            for track_id, boxes in detections_by_id.items():
                name_counts = Counter(b['NAME'] for b in boxes)
                names_considered = set()

                for name in name_counts:
                    relevant_boxes = [b for b in boxes if b['NAME'] == name]

                    # If name is known and either frequent enough or high score
                    if name not in ("No face detected") and (
                        name_counts[name] >= min_freq or
                        any(float(b['SIM_SCORE']) >= min_score for b in relevant_boxes)
                    ):
                        best_box = max(relevant_boxes, key=lambda b: float(b['SIM_SCORE']))
                        final_results.append((
                            best_box['ID'],
                            best_box['FACE'],
                            best_box['DATE'],
                            best_box['TIME'],
                            best_box['SIM_SCORE'],
                            best_box['STATUS'],
                            best_box['NAME'],
                            best_box['movement']
                        ))
                        names_considered.add(name)

                # Optionally include frequent unknowns
                for name in name_counts:
                    if name in names_considered:
                        continue
                    if name in ("Unidentified", "No face detected") and name_counts[name] >= min_freq:
                        unknown_boxes = [b for b in boxes if b['NAME'] == name]
                        best_unknown = max(unknown_boxes, key=lambda b: float(b['SIM_SCORE']))
                        final_results.append((
                            best_unknown['ID'],
                            best_unknown['FACE'],
                            best_unknown['DATE'],
                            best_unknown['TIME'],
                            best_unknown['SIM_SCORE'],
                            best_unknown['STATUS'],
                            best_unknown['NAME'],
                            best_unknown['movement']
                        ))

            return final_results


    def add_movement_to_frames(frames_data):
        id_x_positions = defaultdict(list)
        id_frame_indices = defaultdict(list)

        for idx, frame in enumerate(frames_data):
            for _, boxes in frame.items():
                for box in boxes:
                    id_ = box['ID']
                    x1, _, x2, _ = box['BBOX']
                    center_x = (x1 + x2) / 2
                    id_x_positions[id_].append(center_x)
                    id_frame_indices[id_].append(idx)

        movement_summary = {}
        for id_, x_positions in id_x_positions.items():
            if len(x_positions) < 2:
                movement_summary[id_] = "Insufficient data"
            else:
                delta = x_positions[-1] - x_positions[0]
                direction = "L2R" if delta > 0 else "R2L" if delta < 0 else "No significant movement"
                movement_summary[id_] = f"{direction}-{abs(int(delta))}px"

        updated_frames = copy.deepcopy(frames_data)
        for id_, frames_idx in id_frame_indices.items():
            for idx in frames_idx:
                for key, boxes in updated_frames[idx].items():
                    for box in boxes:
                        if box['ID'] == id_:
                            box['movement'] = movement_summary[id_]

        return updated_frames



    skip_frame_count = 0
    past_face_frames_memory = []
    frame_count_face = 1
    no_of_frame_in_memory = 7

    blue_line_x_cord = 0 
    red_line_x_cord = 0
    height = 0
    fmtt = "%d/%m/%Y %H:%M:%S"   # before formating
    fmt = "%d-%m-%Y %H-%M-%S-%f"  # after formating




    face_folders = "detected faces"
    unknown_faces_folder_path = os.path.join(face_folders,"Unidentified faces")
    identified_face_path = os.path.join(face_folders,"identified faces")

    os.makedirs(face_folders,exist_ok=True)
    os.makedirs(identified_face_path,exist_ok=True)
    os.makedirs(unknown_faces_folder_path,exist_ok=True)

    print(f"Process Started cam id ----->  {cam_id} ✅✅✅")
    while not stop_event.is_set():
        try:
            frame_data = queue_frames.get(block=True,timeout=2) # Reduced timeout for quicker responsiveness
        except queue.Empty:
            print(f"Queue is Empty 👎👎👎👎👎   size is :  {queue_frames.qsize()} : CamID   : {cam_id}")
            # print("waiting process for 0.5 seconds --->  ",cam_id)
            time.sleep(0.5)
            if starting_time is not None:
                break
            else:    
                continue
        
        if frame_data is None: # Producer sent None to signal end
            print(f"[Consumer Process {cam_id}] Received None signal from producer. Stream ended.  ❌❌❌❌")
            # print("Processing Last frame")
            # <<<<<<<<<<<<<<<<<<<< CODE TO BE LAST FRAME  >>>>>>>>>>>>>>>>>>>>
            if len(past_face_frames_memory) >=1:
                past_face_frames_memory = add_movement_to_frames(frames_data=past_face_frames_memory)

                is_crowd = True 
                finalaized_data = final_names(sample=past_face_frames_memory,IsCrowd=is_crowd)
                final_df  = pd.DataFrame([(id,dt,tm,score,status,name,movement) for  id,face,dt,tm,score,status,name,movement  in finalaized_data]).reset_index()
                # print("Last dataframe _____ :   \n",final_df)
                if final_df.empty: 
                    past_face_frames_memory.clear()
                    frame_count_face = 1
                    continue
                persons = final_df[5].unique()

                for person in persons:
                    if not person.lower().startswith('unidentified'):  # identified person
                        filters_df = final_df[final_df[5]== person]
                        idx, tracking_id,dt,tm,score,status,name,movement=filters_df.sort_values(by=3,ascending=False).iloc[0]
                        frame_time_sampt = datetime.strptime(dt + " " + tm, fmt)
                        log_face_info(frame,name,cam_id=cam_id,frame_time=frame_time_sampt,starting_time=starting_time)
                        print("---->  Name  : ",name,"   --->   ",score,"   ----->    ",frame_time_sampt)
                        file_name = f"{name}_{dt}_{tm}_{round(score,2)}.jpg"
                        cv.imwrite(os.path.join(identified_face_path,file_name),finalaized_data[idx][1])  


                    else: 
                        unknown_filters_df = final_df[final_df[5]== "Unidentified"]
                        unique_ids =  unknown_filters_df[0].unique()
                        for id in unique_ids:
                            unknown_person = unknown_filters_df[unknown_filters_df[0] == id]
                            idx,tracking_id,dt,tm,score,status,name,movement=unknown_person.sort_values(by=3,ascending=False).iloc[0]
                            frame_time_sampt = datetime.strptime(dt + " " + tm, fmt)
                            log_face_info(frame,name,cam_id=cam_id,frame_time=frame_time_sampt,starting_time=starting_time)  # FEED FETCH TIME                           
                            print("---->  Name  : ",name,"   --->   ",score,"   ----->    ",frame_time_sampt)
                            file_name = f"{name}_{dt}_{tm}_{round(score,2)}.jpg"
                            cv.imwrite(os.path.join(identified_face_path,file_name),finalaized_data[idx][1])  

            past_face_frames_memory.clear()
            break # Exit loop if producer sent None

        frame, current_msec, frame_count = frame_data
        nparr = np.frombuffer(frame, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        # print("shape of frame : ",frame.shape)
        # print(f"camid : {cam_id} Frame No : {frame_count}    Queue size  :  {queue_frames.qsize()}      ")
        print(f"camid     : {cam_id}  Queue size  :  {queue_frames.qsize()}")


        frame_time = None 
        if starting_time is not None:  # if video is provided 
            fmtt = "%d/%m/%Y %H:%M:%S"
            started_time = datetime.strptime(starting_time,fmtt )  
            frame_time = started_time + timedelta(milliseconds=current_msec)  
        else:
            frame_time = current_msec


        height, width, _ = frame.shape
        blue_line_x_cord = 300
        red_line_x_cord = width -250
        frame,  detected_face_bbox_in_region ,skip = idintifie_person(frame, red_line_x_cord, blue_line_x_cord,frame_time,starting_time)
        
        if skip:
            skip_frame_count += 1
            continue

        if (len(past_face_frames_memory) < no_of_frame_in_memory) and (len(detected_face_bbox_in_region) >= 1):
            past_face_frames_memory.append({f"frame_{frame_count_face}": detected_face_bbox_in_region})
            frame_count_face += 1
        else: # memory filled
            if len(past_face_frames_memory) == no_of_frame_in_memory:
                past_face_frames_memory = add_movement_to_frames(frames_data=past_face_frames_memory) 
                frame_frequency = [len(frame[list(frame.keys())[0]]) for frame in past_face_frames_memory]  # count boxes in frame wise 
                detected_persons_in_hall = int(sum(frame_frequency)/len(frame_frequency))
                is_crowd = False
                if detected_persons_in_hall >4:  # 4 threshold for crowd
                    # crowd in hall, ignore the ID Tracking
                    is_crowd = True 
                    finalaized_data = final_names(sample=past_face_frames_memory,IsCrowd=is_crowd)
                    final_df  = pd.DataFrame([(id,dt,tm,score,status,name,movement) for  id,face,dt,tm,score,status,name,movement  in finalaized_data]).reset_index()
                    # print("first dataframe _____ :   \n",final_df)
                    if final_df.empty: 
                        past_face_frames_memory.clear()
                        frame_count_face = 1
                        continue
                    persons = final_df[5].unique()
                    for person in persons:

                        if not person.lower().startswith('unidentified'):  # identified person
                            # print("<<<<<<<<<<<<<<< crowd known   >>>>>>>>>>>>>>>>>>>>>>>>>>>")
                            filters_df = final_df[final_df[5]== person]
                            idx, tracking_id,dt,tm,score,status,name,movement=filters_df.sort_values(by=3,ascending=False).iloc[0]
                            frame_time_sampt = datetime.strptime(dt + " " + tm, fmt)
                            log_face_info(frame,name,cam_id=cam_id,frame_time=frame_time_sampt,starting_time=starting_time)
                            print("---->  Name  : ",name,"   --->   ",score,"   ----->    ",frame_time_sampt)
                            file_name = f"{name}_{dt}_{tm}_{round(score,2)}.jpg"
                            cv.imwrite(os.path.join(identified_face_path,file_name),finalaized_data[idx][1])   
                        
                        else:  # uknown person
                            # print("<<<<<<<<<<<<<<< crowd unknown   >>>>>>>>>>>>>>>>>>>>>>>>>>>")  # we did not considered uknown in crowd 
                            unknown_filters_df = final_df[final_df[5]== "Unidentified"]
                            unique_ids =  unknown_filters_df[0].unique()
                            for id in unique_ids:
                                unknown_person = unknown_filters_df[unknown_filters_df[0] == id]
                                idx,tracking_id,dt,tm,score,status,name,movement=unknown_person.sort_values(by=3,ascending=False).iloc[0]
                                frame_time_sampt = datetime.strptime(dt + " " + tm, fmt)
                                log_face_info(frame,name,cam_id=cam_id,frame_time=frame_time_sampt,starting_time=starting_time)  # FEED FETCH TIME
                                print("---->  Name  : ",name,"   --->   ",score,"   ----->    ",frame_time_sampt)
                                file_name = f"{name}_{dt}_{tm}_{round(score,2)}.jpg"
                                cv.imwrite(os.path.join(identified_face_path,file_name),finalaized_data[idx][1])  

                    past_face_frames_memory.clear()
                    frame_count_face = 1
                
                else:  # NON CROWD

                    finalaized_data = final_names(sample=past_face_frames_memory,IsCrowd=is_crowd)
                    final_df  = pd.DataFrame([(id,dt,tm,score,status,name,movement) for  id,face,dt,tm,score,status,name,movement  in finalaized_data]).reset_index()
                    # print("second dataframe _____ :   \n",final_df)
                    if final_df.empty: 
                        past_face_frames_memory.clear()
                        frame_count_face = 1
                        continue

                    persons = final_df[5].unique()  
                    for person in persons:
                        
                        
                        if not person.lower().startswith('unidentified'):  # identified person
                            # print("<<<<<<<<<<<<<<<  known   >>>>>>>>>>>>>>>>>>>>>>>>>>>")
                            filters_df = final_df[final_df[5]== person]
                            idx,tracking_id,dt,tm,score,status,name,movement=filters_df.sort_values(by=3,ascending=False).iloc[0]  # sorted based on similarity score                                                      
                            frame_time_sampt = datetime.strptime(dt + " " + tm, fmt)        
                            log_face_info(frame,name,cam_id=cam_id,frame_time=frame_time_sampt,starting_time=starting_time)
                            print("---->  Name  : ",name,"   --->   ",score,"   ----->    ",frame_time_sampt)
                            file_name = f"{name}_{dt}_{tm}_{round(score,2)}.jpg"
                            cv.imwrite(os.path.join(identified_face_path,file_name),finalaized_data[idx][1])  
                        
                        
                        else:  # uknown person
                            # print("<<<<<<<<<<<<<<<  unknown   >>>>>>>>>>>>>>>>>>>>>>>>>>>")
                            unknown_filters_df = final_df[final_df[5]== "Unidentified"]   
                            unique_ids =  unknown_filters_df[0].unique()
                            for id in unique_ids:
                                unknown_person = unknown_filters_df[unknown_filters_df[0] == id]
                                idx,tracking_id,dt,tm,score,status,name,movement=unknown_person.sort_values(by=3,ascending=False).iloc[0]                                   
                                frame_time_sampt = datetime.strptime(dt + " " + tm, fmt)
                                log_face_info(frame,name,cam_id=cam_id,frame_time=frame_time_sampt,starting_time=starting_time) 

                                print("---->  Name  : ",name,"   --->   ",score,"   ----->    ",frame_time_sampt)                                     
                                file_name = f"{name}_{dt}_{tm}_{round(score,2)}.jpg"
                                cv.imwrite(os.path.join(unknown_faces_folder_path,file_name),finalaized_data[idx][1])                     

                    past_face_frames_memory.clear()
                    frame_count_face = 1

            if len(detected_face_bbox_in_region) >= 1:  # MEMORY FULL, AND OPERATION DONE BUT DETECTED FACES HAS BOXES THAT WE R KEEPING
                past_face_frames_memory.append({f"frame_{frame_count_face}": detected_face_bbox_in_region})
                frame_count_face += 1

        
            
        # else:
        #     pass
        # frame = cv.resize(frame, (1920,1080))   # 2000,1800
        # cv.line(frame, (blue_line_x_cord, 0), (blue_line_x_cord, height), color=(0, 255, 0), thickness=2)
        # cv.line(frame, (red_line_x_cord, 0), (red_line_x_cord, height), color=(0, 0, 255), thickness=2)
        # cv.imshow(F"Streaming Process {cam_id} ", frame)
        # if cv.waitKey(27) & 0xFF == ord('q'):
        #     break

    ended_time = (time.time()) 
    print(f"Execution Time {(ended_time- start)/60:.2f} seconds.")



# ==== MULTI-CAMERA ENTRY POINT ====
if __name__ == "__main__":
    camid_api_url = r"http://192.168.0.48:1073/addmodifycamerasettings/get-by-url?rtspUrl=" 
    java_1 = "rtsp://admin:12345@192.168.29.25:554/cam/realmonitor?channel=1&subtype=0"
    biometric = "rtsp://admin:admin@192.168.29.3:554/unicaststream/1"
    pm_cabin = "rtsp://admin:admin@192.168.29.6:554/unicaststream/1"
    bathroom = "rtsp://admin:admin@123@192.168.29.56:554/unicaststream/1"
    mobile_recording,mbl_rectmstmp = r"\\SAGNAS\java\Ranjit Singh\AI_Project Video 01052025\rotated_video-1.mp4","22/06/2025 00:00:00"
    live_exit_link = r"rtsp://admin:admin@192.168.29.121:554/unicaststream/1"  
    live_entry_link = r"rtsp://admin:Matrix@554@192.168.29.124:554/unicaststream/1" 
    recording_17_july_punchout , tmstmp0073 = r"\\SAGNAS\java\Jatin\AI Project Video\recording_17_punchout.AVI" , "17/07/2025 18:55:51"
    recording_10_july_after_lunch , b  = r"\\SAGNAS\java\Ranjit Singh\AI_Project Video 01052025\recording_10_july_after_lunch.AVI","10/07/2025 13:55:28"
    recording_8_aug_after_lunch , tmstmp  = r"\\SAGNAS\java\Jatin\videos\DOWNLOAD_06082025_135459_C2.AVI","06/08/2025 13:54:58"
    recording_6_aug_punchout ,tmstmp2= r"C:\MatrixDeviceClient\QuickBackup\00_1b_09_10_a6_38\192.168.29.121\DOWNLOAD_06082025_185525_C1.AVI","06/08/2025 18:55:24"
    mrng_8_augst ,tmstmp3= r"C:\MatrixDeviceClient\QuickBackup\00_1b_09_10_a6_38\MATRIX COMSEC\DOWNLOAD_08082025_092410_C2.AVI","08/08/2025 09:24:08"
    recording_20_m1,tmstmp02=r"\\sagnas\java\Jatin\videos\20_1.AVI","20/08/2025 09:20:26"
    recording_20_m2,tmstmp03=r"\\sagnas\java\Jatin\videos\20_2.AVI","20/08/2025 09:40:26"

    rtsp_urls = [
    
              live_entry_link,live_exit_link
             
             ]
    # 
    recorded_urls = [
     mobile_recording
        
    ]
    recorded_video_time_stamp = [ mbl_rectmstmp]


    all_processes = []
    all_stop_events = [] 
    all_queues_for_monitoring = []

    if rtsp_urls:
        print("Live streaming Process initiated...")
        for idx, rtsp_url in enumerate(rtsp_urls):
            q = Queue(maxsize=2000) 
            e = Event()           
            
            all_stop_events.append(e) 

            producer_thread = threading.Thread(
                target=frame_reader,    
                args=(rtsp_url, idx, q, e),    # video_path, cam_id, frame_queue, stop_event, starting_time=None
                name=f"FrameReader-Cam{idx}",
                daemon=True
            )
            producer_thread.start()
            all_queues_for_monitoring.append(q) 
        
            print("Wait for 2s to start the Consumer part")
            time.sleep(2) 
        
            p1 = Process(target=process_camera_stream, args=(q, idx, None, e))  # queue_frames, cam_id, starting_time=None, stop_event=None
            p2 = Process(target=process_camera_stream, args=(q, idx, None, e))
            p1.start()
            p2.start()
            all_processes.append(p1)
            all_processes.append(p2)

        try:
            for p in all_processes:
                p.join()
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Stopping all threads and processes.")
            for e in all_stop_events:
                e.set()

            for p in all_processes:
                if p.is_alive():
                    p.terminate()
                    p.join()

            print("All threads and processes terminated successfully.")


        stop = time.time()
        print(f"processed time : {(stop-start)/60:.2f}.")
        print("process completed! cameID  👍   ",idx )


    else:
        print("Video streaming Process initiated...")
        processes = []
        for idx,( url,starttime) in enumerate(zip(recorded_urls,recorded_video_time_stamp)):  
            q = Queue(maxsize=0) 
            e = Event()           
            all_stop_events.append(e) 
            


            producer_thread = threading.Thread(target=frame_reader,args=(url, idx, q, e,starttime),
            name=f"FrameReader-Cam{idx}",
            daemon=True
            )
            producer_thread.start()
            all_queues_for_monitoring.append(q) 
            print("Wait for 1s to start the Consumer part")
            time.sleep(1)
            p1 = Process(target=process_camera_stream, args=(q,idx,starttime,e) )     #queue_frames, cam_id, starting_time=None, stop_event=None
            p2 = Process(target=process_camera_stream, args=(q,idx,starttime,e) )    

            p1.start()
            p2.start()

            processes.append(p1)
            processes.append(p2)




        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Stopping all threads and processes.")
            for e in all_stop_events:
                e.set()


            for p in all_processes:
                if p.is_alive():
                    p.terminate()
                    p.join()

            print("All threads and processes terminated successfully.")
        stop = time.time()
        print(f"processed time : {(stop-start)/60:.2f}.")
        print("process completed! cameID  👍   ",idx )



  




 
