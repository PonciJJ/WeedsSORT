import cv2
import torch
import argparse
from tracker.tracker import Tracker, bbox_to_meas, meas_to_mot
from tools.visualizer import Visualizer
from yolov11.ultralytics import YOLO
import tools.data_manager as DataUtils
# from SuperGluePretrainedNetwork.models.matching import Matching
import time
import numpy as np


def parse_opt():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Example script to demonstrate command line options.')
    # Add options
    # parser.add_argument('-s', '--source', type=str, default='/media/nvidia/Elements/data/MOT/LettuceMOT/straight1/img/', help='Source of files (dir, file, video, ...)')
    parser.add_argument('-s', '--source', type=str, default='/media/nvidia/Elements/data/test_data/bf1/', help='Source of files (dir, file, video, ...)')
    parser.add_argument('-o', '--output', type=str, default='runs/', help='Output file path')
    # parser.add_argument('-w', '--weights', type=str, default='/media/nvidia/Elements/data/weights/0007.pt', help='Weights of YOLOv5 detector')
    # parser.add_argument('-w', '--weights', type=str, default='/home/nvidia/best.pt', help='Weights of YOLOv5 detector')
    parser.add_argument('-w', '--weights', type=str, default='./best.pt', help='Weights of YOLOv5 detector')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--features', type=str, default="optical_flow", help='Features for camera motion compensation (orb, optical_flow, superglue...)')
    parser.add_argument('--transform', type=str, default="affine", help='Tranformation for estimation of camera motion')
    parser.add_argument('-v', '--visualize', type=bool, default=True, help='Enable or disable real-time visualization')
    opt = parser.parse_args()
    return opt


def main(opt):
    # model = torch.hub.load('./yolov5', 'custom', path=opt.weights, source="local")
    # model.conf = opt.conf_thres
    import pdb;pdb.set_trace()
  
    # model1 = YOLO("/home/nvidia/mot/WeedsSORT/yolo11s.pt")
    model1 = YOLO("/home/nvidia/mot/WeedsSORT/yolov11/runs/detect/train5/weights/best.pt")

    # Initialize tracker
    tracker = Tracker(features=opt.features, transform=opt.transform)

    # If visualizer, initialize visualizer
    print(opt.visualize)
    if opt.visualize:
        visualizer = Visualizer()

    # Create folder to save results
    folder_path = DataUtils.create_experiment_folder()
    open(folder_path + "/WeedsSORT.txt", 'w').close()

    # Load data and start tracking
    # import pdb;pdb.set_trace()
    dataset = DataUtils.DataLoader(opt.source)


    for i, frame in dataset:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d_start = time.time()
        #1920 for weeds and 1280 for lettuces
        # pred = model(gray_image, size=1280)
        pred1 = model1(frame)
        d_time = (time.time() - d_start) * 1000
        if i == 1:
            for box in pred1[0].boxes.xyxy[pred1[0].boxes.cls == 1]:
            # for box in pred1[0].boxes.xyxy:  
                prev_image = gray_image
                tracker.add_track(1, bbox_to_meas(box.cpu().detach().numpy()), 0.05, 0.00625)
        else:
            c_start = time.time()
            Aff = tracker.camera_motion_computation(prev_image, gray_image)
            c_time = (time.time() - c_start) * 1000
            prev_image = gray_image.copy()
            t_start = time.time()
            #cls == 1 for weed
            # tracker.update_tracks(pred1[0].boxes.xyxy[pred1[0].boxes.cls == 1].cpu().detach().numpy(), Aff, frame)
            tracker.update_tracks(pred1[0].boxes.xyxy.cpu().detach().numpy(), Aff, frame)
            # tracker.update_tracks(pred.xyxy[0].cpu().detach().numpy(), Aff, frame)
            t_time = (time.time() - t_start) * 1000
            for track in tracker.tracks:
                if opt.visualize and track.display:
                    frame = visualizer.draw_track(track, frame)
                with open(folder_path + "/WeedsSORT.txt", 'a') as f:
                    mot = meas_to_mot(track.x)
                    temp = str(mot[0]) + ', ' + str(mot[1]) + ', ' + str(mot[2]) + ', ' + str(mot[3])
                    f.write(str(i-1) + ', ' + str(track.id) + ', ' + temp + ', -1, -1, -1, -1' + '\n')
            if opt.visualize:
                # import pdb;pdb.set_trace()
                # visualizer.display_image(frame, 0)
                cv2.imwrite("/media/nvidia/Elements/data/test_data/output/bf1/" + str(i).zfill(5) + ".jpg", frame)
            # Terminal output
            print("Frame {}/{} || Detections {} ({:.2f} ms) || Camera Correction ({:.2f} ms) || Tracking {} ({:.2f} ms)".format(
                i, dataset.len, int(len(pred1[0].boxes.xyxy)), d_time, c_time, len(tracker.tracks), t_time))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
