import cv2
from cv2 import ORB_HARRIS_SCORE
import numpy as np
import ffmpeg

def vidwrite(fn, images, bitrate, framerate=60, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width,channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=framerate)
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, **{'b:v': bitrate })
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    print("Reading in original mp4")

    videoin = "test.mp4"

    cap = cv2.VideoCapture(videoin)
    probe = ffmpeg.probe(videoin)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    original_bitrate = int(video_stream["bit_rate"])
    
    video = []
    while(True):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)

    print("Video read in done...")
    video = np.array(video)
    bit_rates = [original_bitrate, original_bitrate/2, original_bitrate/3, original_bitrate/4]

    for br in bit_rates:
        print(br)
        videoout = "out_%i.mp4"%(br)
        new_video_clip_overlay = []
        for frame in video:
            new_frame = np.copy(frame)
            new_frame = cv2.putText(new_frame, "bitrate: %i"%(br), (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), thickness=2)
            new_video_clip_overlay.append(new_frame)
        vidwrite(videoout, new_video_clip_overlay, framerate=60, vcodec='libx264', bitrate=br)
