from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np

from opts import opts
from detectors.detector_factory import detector_factory

import pdb

image_ext = ["jpg", "jpeg", "png", "webp"]
video_ext = ["mp4", "mov", "avi", "mkv"]
time_stats = ["tot", "load", "pre", "net", "dec", "post", "merge"]


def demo(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == "webcam" or opt.demo[opt.demo.rfind(".") + 1 :].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == "webcam" else opt.demo)
        if opt.output_dir != "none":
            if opt.demo == "webcam":
                save_path = os.path.join(opt.output_dir, "webcam.mp4")
            else:
                _, filename_w_ext = os.path.split(opt.demo)
                filename_wo_ext, _ = os.path.splitext(filename_w_ext)
                save_path = os.path.join(opt.output_dir, filename_wo_ext + "_infer.mp4")
        detector.pause = False
        videowriter_initialised = False
        while True:
            success, img = cam.read()
            if not success:
                print("Can't receive frame...")
                break
            # cv2.imshow("input", img)
            ret = detector.run(img)
            if opt.output_dir != "none":
                inferred_img = ret["output_images"][opt.task]
                if not videowriter_initialised:
                    frame_h, frame_w, _ = inferred_img.shape
                    VideoWriter = save_video((frame_h, frame_w), save_path, fps=30)
                    videowriter_initialised = True
                VideoWriter.write(inferred_img)
            time_str = ""
            for stat in time_stats:
                time_str = time_str + "{} {:.3f}s |".format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                if opt.output_dir != "none":
                    VideoWriter.release()
                    print("Video saved successfully")
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind(".") + 1 :].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        for image_name in image_names:
            print("Press Esc to stop, any other key to continue to next image...")
            ret = detector.run(image_name)
            if opt.output_dir != "none":
                inferred_img = ret["output_images"][opt.task]
                _, filename = os.path.split(image_name)
                filename_wo_ext, ext = os.path.splitext(filename)
                new_filename = filename_wo_ext + "_infer" + ext
                save_path = os.path.join(opt.output_dir, new_filename)
                cv2.imwrite(save_path, inferred_img)
            time_str = ""
            for stat in time_stats:
                time_str = time_str + "{} {:.3f}s |".format(stat, ret[stat])
            print(time_str)


def save_video(frame_size: tuple, save_path: str, fps=30):
    height, width = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(save_path, fourcc, fps, (width, height))


if __name__ == "__main__":
    opt = opts().init()
    demo(opt)
