import sys
sys.path.append('core')

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

def vizualize_flow(img, flo, save, counter):
    # convert CWH to WHC format and change device if necessary  
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
 
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)
 
    # concatenate, save and show images
    img_flo = np.concatenate([img, flo], axis=0)
    if save:
        cv2.imwrite(f"demo_frames/infer_frame_{str(counter)}.jpg", img_flo)
    
    cv2.imshow("Optical Flow", img_flo / 255.0)
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    # parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--video', help="path to the video file")
    # parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, weights_only=True))

    model.cuda()
    model.eval()

    if torch.cuda.is_available():
        device = "cuda"
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
        # img = torch.from_numpy(img).permute(2, 0, 1).float().cuda()
        # return img
        
        # frame = cv2.resize(frame, (1024, 512))  # Adjust dimensions as needed
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        t_frame = t_frame.unsqueeze(0).cuda()  # Add batch dimension
        return t_frame

    with torch.no_grad():
        counter = 0
        cap = cv2.VideoCapture(args.video)
        frame_1 = None
        
        while True:
            ret, frame_2 = cap.read()
            # frame_2 = torch.from_numpy(frame_2).permute(2, 0, 1).float().cuda()
            frame_2 = preprocess(frame_2)
            if not ret:
                break
            
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                break
    
            if frame_1 is not None:
                # preprocessing
                # frame_2 = frame_preprocess(frame_2, device)
                # predict the flow
                _flow_diff, flow_up = model.module(frame_1, frame_2, iters=20, test_mode=True)
                ret = vizualize_flow(frame_1, flow_up, save=False, counter=counter)
            
            frame_1 = frame_2
            counter += 1

        cap.release()


if __name__ == "__main__":
    main()


# python inference.py --model=./models/raft-sintel.pth --video ./videos/crowd.mp4