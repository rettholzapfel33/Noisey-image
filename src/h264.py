import os
import cv2
import glob
import ffmpeg
import numpy as np
from PIL import Image

class H264:

    def vidwrite(self, fn, images, bitrate, framerate=60, vcodec='libx264'):
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

                # Check if compression is done
                process.stdin.write(
                    frame
                        .astype(np.uint8)
                        .tobytes()
                )
            process.stdin.close()
            process.wait()

    def convert_mp4_to_jpgs(self, path):

        print("Converting to JPGs")
        video_capture = cv2.VideoCapture(path)
        still_reading, image = video_capture.read()
        frame_count = 0
        while still_reading:
            cv2.imwrite(f"vids/convert/frame_{frame_count:03d}.jpg", image)
            
            # read next image
            still_reading, image = video_capture.read()
            frame_count += 1

    def make_gif(self, frame_folder, name):

        print("Converting to GIFS")
        images = glob.glob(f"{frame_folder}/*.jpg")
        images.sort()
        frames = [Image.open(image) for image in images]
        frame_one = frames[0]
        frame_one.save("vids/gifs/" + str(name) + '.gif', format="GIF", append_images=frames,
                    save_all=True, duration=50, loop=0)


    def displayVideo(self):

        videoin = "vids/default/test.mp4"

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
        bit_rates = [original_bitrate/2]

        for br in bit_rates:
            print(br)
            videoout = "vids/compressed/out_%i.mp4"%(br)
            new_video_clip_overlay = []
            for frame in video:
                new_frame = np.copy(frame)
                new_frame = cv2.putText(new_frame, "bitrate: %i"%(br), (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), thickness=2)
                new_video_clip_overlay.append(new_frame)
                self.vidwrite(videoout, new_video_clip_overlay, framerate=60, vcodec='libx264', bitrate=br)

            self.convert_mp4_to_jpgs(videoout)
            self.make_gif('vids/convert')

def main():
    
    videoin = "vids/default/test.mp4"

    h264 = H264()

    i = 0
    name = ''

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
    bit_rates = [original_bitrate, original_bitrate/4]

    for br in bit_rates:

        videoout = "vids/compressed/out_%i.mp4"%(br)
        new_video_clip_overlay = []
        for frame in video:
            new_frame = np.copy(frame)
            new_frame = cv2.putText(new_frame, "bitrate: %i"%(br), (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), thickness=2)
            new_video_clip_overlay.append(new_frame)
        h264.vidwrite(videoout, new_video_clip_overlay, framerate=60, vcodec='libx264', bitrate=br)

        if i == 0: 
            name = 'original'
        else: 
            name = 'compressed'

        h264.convert_mp4_to_jpgs(videoout)
        h264.make_gif('vids/convert', name)

        # Remove all jpgs
        for f in os.listdir('vids/convert'): os.remove(os.path.join('vids/convert', f))
        for f in os.listdir('vids/compressed'): os.remove(os.path.join('vids/compressed', f))

        # Increment 
        i = i + 1