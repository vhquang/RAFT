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

def vizualize_flow(img: torch.Tensor, flo, save, counter):
    # convert CHW (used by model) to HWC (used by cv2) format and change device if necessary  
    img_hwc = img[0].permute(1, 2, 0).cpu().numpy()
    flo_hwc = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo_img = flow_viz.flow_to_image(flo_hwc)
    flo_img = cv2.cvtColor(flo_img, cv2.COLOR_RGB2BGR)
 
    if save:
        # concatenate, save and show images
        concat = np.concatenate([img_hwc, flo_img], axis=0)
        outfile = f"output/infer_frame_{str(counter)}.jpg"
        cv2.imwrite(outfile, concat)
        # cv2.imshow("Optical Flow", flo_img / 255.0)
        print(f"Image saved to {outfile}")
    else:
        cv2.imshow("Optical Flow", flo_img / 255.0)
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    # parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--video', help="path to the video file")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

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
        cap = cv2.VideoCapture(args.video)
        frame_1 = None
        
        while True:
            ret, raw_frame = cap.read()
            
            if not ret:
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            frame_2 = preprocess(raw_frame)
    
            if frame_1 is not None:
                # predict the flow
                _flow_diff, flow_up = model.module(frame_1, frame_2, iters=20, test_mode=True)
                if counter %10 == 0:
                    _ret = vizualize_flow(frame_2, flow_up, save=True, counter=counter)
            
            frame_1 = frame_2
            counter += 1

        cap.release()


if __name__ == "__main__":
    main()


# python inference.py --model=./models/raft-sintel.pth --video ./videos/crowd.mp4