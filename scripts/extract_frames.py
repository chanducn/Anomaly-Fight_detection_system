import os 
import cv2

def extract_frames_V(video_path,output_dir,max_frames=16):
    os.makedirs(output_dir,exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval  = max(1 , total_frames // max_frames)


    count = 0  #Track frame number
    saved = 0 #Track number of saved frames

    # looping through video and save frames

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break

        # saving frame every time interval

        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir,f"frame_{saved}.jpg")
            cv2.imwrite(frame_path,frame)  # save the frame as image
            saved +=1

        count += 1 # move to the next frame
    
    cap.release()


def process_data(input_base_dir, output_base_dir):
    class_map = {
        'Fight': 'Fight',
        'NonFight': 'NonFight'
    }

    for split in ['train', 'val']:
        for folder in os.listdir(os.path.join(input_base_dir, split)):
            print(f"Found folder: {folder}")
            class_name = class_map.get(folder)
            if class_name is None:
                print(f"Skipping folder: {folder}")
                continue

            folder_path = os.path.join(input_base_dir, split, folder)
            output_folder = os.path.join(output_base_dir, split, class_name)
            os.makedirs(output_folder, exist_ok=True)

            for idx, video in enumerate(os.listdir(folder_path)):
                video_path = os.path.join(folder_path, video)
                clip_output = os.path.join(output_folder, f"{folder.replace(' ', '')}_{idx}")
                extract_frames_V(video_path, clip_output)

                # Example usage to process everything
process_data("dataset", "processed_data")
