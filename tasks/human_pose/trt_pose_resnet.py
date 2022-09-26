import json
import trt_pose.coco
import trt_pose.models
import torch
from torch2trt import TRTModule
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import ipywidgets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg
import pyrealsense2 as rs
import numpy as np

MODEL_WIDTH = 224
MODEL_HEIGHT = 224#对应resenet18
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

def preprocess(image):
    global device
    device = torch.device('cuda')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def execute(image):
    image = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT))
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    image = cv2.resize(image, (CAPTURE_WIDTH, CAPTURE_HEIGHT))###回复原本分辨率
    return image

if __name__ == '__main__':
    
    with open('human_pose.json', 'r') as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    print("success open the json")
    data = torch.zeros((1, 3, MODEL_HEIGHT, MODEL_WIDTH)).cuda()
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'


    # ###优化模型步骤，只需运行一次，后面直接读取保存的优化后模型即可
    # import trt_pose.models
    # num_parts = len(human_pose['keypoints'])
    # num_links = len(human_pose['skeleton'])
    # model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    # import torch
    # MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
    # model.load_state_dict(torch.load(MODEL_WEIGHTS))
    # import torch2trt
    # ##优化模型
    # model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    # #这一步是把优化后的模型保存，只需要运行一次，后面直接读取保存的优化后模型即可
    # torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)


    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
    print("success load the optimized model")

    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)


    print("start estimate humanpose")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, CAPTURE_WIDTH, CAPTURE_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, CAPTURE_WIDTH, CAPTURE_HEIGHT, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)#声明与谁对齐-与颜色流（align_to=rs.stream.color）
    try:
        while True:
            t_start = time.time()
            torch.cuda.current_stream().synchronize()
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)#声明谁要对齐-采集获得的frames
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            # Get aligned image
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            #color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            image_pose = execute(color_image)###Key step
            torch.cuda.current_stream().synchronize()
            t_end = time.time()
            t_oneframe = t_end - t_start
            #count = count + 1
            cv2.namedWindow('Pose', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Pose', image_pose)
            cv2.waitKey(1)

            FPS = 1/t_oneframe
            print("FPS:" + str(FPS))
    finally:
        pipeline.stop()



