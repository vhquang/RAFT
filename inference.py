from __future__ import annotations

import sys
sys.path.append('core')
from pathlib import Path

import argparse
import numpy as np

import cv2
import torch

from raft import RAFT
from utils import flow_viz

# def inference(args):
#     model = RAFT(args)
#     # load pretrained weights
#     pretrained_weights = torch.load(args.model, weights_only=True)

def get_cpu_model(model):
    raise NotImplementedError
    new_model = OrderedDict()
    # get all layer's names from model
    for name in model:
        # create new name and update new model
        new_name = name[7:]
        new_model[new_name] = model[name]
    return new_model


def vizualize_flow(img: torch.Tensor, flo, video_path: Path, counter) -> np.ndarray:
    # convert CHW (used by model) to HWC (used by cv2) format and change device if necessary  
    img_hwc = img[0].permute(1, 2, 0).cpu().numpy()
    flo_hwc = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to BGR image
    flo_img = flow_viz.flow_to_image(flo_hwc, convert_to_bgr=True)
    # flo_img = cv2.cvtColor(flo_img, cv2.COLOR_RGB2BGR)
 
    # concatenate, save and show images
    concat = np.concatenate([img_hwc, flo_img], axis=0)
    outfile = f"output/infer_frame{video_path.stem}_{str(counter)}.jpg"
    # cv2.imwrite(outfile, concat)
    # cv2.imshow("Optical Flow", flo_img)
    # print(f"Image saved to {outfile}")
    return concat


def save_video(frames: list[np.ndarray], video_path: Path) -> Path:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    h, w, _c = frames[0].shape
    print(f"Video dimensions: {w}x{h}, {fps} fps")

    video_output = f'output/video/raft_{video_path.stem}.mp4'
    writer = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        writer.write(frame.astype(np.uint8))
    writer.release()
    print(f"Video saved to {video_output}")
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    # parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--video', help="path to the video file")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    video_path = Path(args.video)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.cuda()
    model.eval()

    # if torch.cuda.is_available():
    #     device = "cuda"
    #     # parallel between available GPUs
    #     model = torch.nn.DataParallel(model)
    #     # load the pretrained weights into model
    #     model.load_state_dict(pretrained_weights)
    #     model.to(device)
    # else:
    #     raise RuntimeError("CUDA is not available")
    #     device = "cpu"
    #     # change key names for CPU runtime
    #     pretrained_weights = get_cpu_model(pretrained_weights)
    #     # load the pretrained weights into model
    #     model.load_state_dict(pretrained_weights)

    def preprocess(frame: np.ndarray) -> torch.Tensor:
        # frame = cv2.resize(frame, (1024, 512))  # Adjust dimensions as needed
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # permute to change from HWC to CHW format.
        t_frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        t_frame = t_frame.unsqueeze(0).cuda()  # Add batch dimension
        return t_frame

    with torch.no_grad():
        counter = 1
        cap = cv2.VideoCapture(str(video_path))
        frame_1 = None
        result_frames = []
        
        while True:
            ret, raw_frame = cap.read()
            
            if not ret:
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            frame_2 = preprocess(raw_frame)
            if frame_1 is None:
                frame_1 = frame_2
                counter += 1
                continue
    
            # predict the flow
            _flow_diff, flow_upsample = model.module(frame_1, frame_2, iters=2, test_mode=True)
            combine_frame = vizualize_flow(frame_2, flow_upsample, video_path=video_path, counter=counter)
            result_frames.append(combine_frame)
            
            frame_1 = frame_2
            counter += 1
            print(f"Processed frame {counter}")

            # if counter == 5:
            #     break

        cap.release()
        cv2.destroyAllWindows()

        save_video(result_frames, video_path)


if __name__ == "__main__":
    main()


# python inference.py --model=./models/raft-sintel.pth --video ./videos/crowd.mp4