import time
from multiprocessing import Process

import cv2
import pyttsx3
import numpy as np

video_path = 'sample.mp4'
# text = ""

eng = pyttsx3.init()
def speak(text):
    # global text
    # Engine created
    # time.sleep(2)

    print(text)
    eng.say(text)
    # Runs for small duration of time ohterwise we may not be able to hear
    eng.runAndWait()


def play_video():
    # global old_caption, new_caption
    cap = cv2.VideoCapture(video_path)
    print("running")
    next_frame = 0
    # processes = []
    # dict = {"checkpoint_path": checkpoint_path,"image_features_extract_model": image_features_extract_model,
            # "top_k": top_k, "tokenizer": tokenizer}
    # with open('global_state.pickle', 'wb') as handle:
        # pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    next_frame = 0

    while (cap.isOpened()):
        next_frame = next_frame + 1
        print("player:"+str(next_frame))
        if next_frame == 160:
            next_frame = 0
        #print("running loop")
        ret, frame = cap.read()
        imS = cv2.resize(frame, (530, 300))
        cv2.imshow('frame', imS)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# if __name__ == '__main__':

# speak("i am python")




