import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import sys
import cv2
import numpy as np
import open3d as o3d
import time
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from argparse import ArgumentParser
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from utils.graphics_utils import focal2fov
from scene.shared_objs import SharedCam, SharedGaussians, SharedPoints, SharedTargetPoints
from gaussian_renderer import render, network_gui
from scene import GaussianModel

from mp_Tracker import Tracker
from mp_Mapper import Mapper
# from mp_Sender import Sender

from multiprocessing import shared_memory, Manager

torch.multiprocessing.set_sharing_strategy('file_system')


class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug
        
class GS_ICP_SLAM(SLAMParameters):
    def __init__(self, args):

        super().__init__()
        self.dataset_path = args.dataset_path
        self.config = args.config
        self.output_path = args.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = args.verbose
        self.keyframe_th = float(args.keyframe_th)
        self.knn_max_distance = float(args.knn_maxd)
        self.overlapped_th = float(args.overlapped_th)
        self.max_correspondence_distance = float(args.max_correspondence_distance)
        self.trackable_opacity_th = float(args.trackable_opacity_th)
        self.overlapped_th2 = float(args.overlapped_th2)
        self.save_results = args.save_results
        
        config_file = open(self.config)
        config_file_ = config_file.readlines()

        self.camera_parameters = config_file_[2].split()
        self.W = int(self.camera_parameters[0])
        self.H = int(self.camera_parameters[1])
        self.fx = float(self.camera_parameters[2])
        self.fy = float(self.camera_parameters[3])
        self.cx = float(self.camera_parameters[4])
        self.cy = float(self.camera_parameters[5])
        self.depth_scale = float(self.camera_parameters[6])
        self.depth_trunc = float(self.camera_parameters[7])

        self.estimation_parameters = config_file_[6].split()
        self.Std_t = float(self.estimation_parameters[0])
        self.Std_R = float(self.estimation_parameters[1])
        self.alpha = float(self.estimation_parameters[2])
        self.epsilon = float(self.estimation_parameters[3])
        self.beta = float(self.estimation_parameters[4])
        self.lambda_0 = float(self.estimation_parameters[5])
        self.depth_noise = float(self.estimation_parameters[6])

        self.optimizer = config_file_[10].split()
        self.opt_mode = int(self.optimizer[0])

        self.player = config_file_[14].split()
        self.viewer = int(self.player[0])
        self.play_mode = int(self.player[1])
        if self.viewer:
            rr.init("3dgsviewer")
            rr.spawn(connect=False)

        self.image_index = config_file_[18].split()
        self.start_idx = float(self.image_index[0])
        self.end_idx = float(self.image_index[1])
        self.len_idx = int(self.end_idx - (self.start_idx - 1))

        self.options = config_file_[22].split()
        self.downsample_size = int(self.options[0])
        self.topic_num = int(self.options[1])
        self.max_fps = float(self.options[2])

        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_size)
        
        # Shared memory 핸들러
        self.shared_memories = []
        for i in range(self.topic_num):
            shm_name = f"img_rendered_{i}"
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                self.shared_memories.append(shm)
                print(f"Connected to shared memory: {shm_name}")
            except FileNotFoundError:
                print(f"Shared memory {shm_name} not found. Ensure it is created by the producer.")
                self.shared_memories.append(None)

        if all(shm is None for shm in self.shared_memories):
            raise RuntimeError("No shared memory segments available. Ensure the producer process is running.")
        # Shared memory 핸들러

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        
        # Make test cam
        # To get memory sizes of shared_cam
        test_rgb_img, test_depth_img = self.get_init_image()
        test_points, _, _, _ = self.downsample_and_make_pointcloud(test_depth_img, test_rgb_img)

        # Get size of final poses
        # len_idx = len(self.trajmanager.gt_poses)
        
        
        # Shared objects
        self.shared_cam = SharedCam(FoVx=focal2fov(self.fx, self.W), FoVy=focal2fov(self.fy, self.H),
                                    image=test_rgb_img, depth_image=test_depth_img,
                                    cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy)
        self.shared_new_points = SharedPoints(test_points.shape[0])
        self.shared_new_gaussians = SharedGaussians(test_points.shape[0])
        self.shared_target_gaussians = SharedTargetPoints(10000000)
        self.end_of_dataset = torch.zeros((1)).int()
        self.is_tracking_keyframe_shared = torch.zeros((1)).int()
        self.is_mapping_keyframe_shared = torch.zeros((1)).int()
        self.target_gaussians_ready = torch.zeros((1)).int()
        self.new_points_ready = torch.zeros((1)).int()
        self.final_T_GTs = torch.zeros((self.len_idx,4,4)).float()
        self.final_T_opts = torch.zeros((self.len_idx,4,4)).float()

        self.image_r = torch.zeros((self.len_idx,self.H,self.W)).float()
        self.image_g = torch.zeros((self.len_idx,self.H,self.W)).float()
        self.image_b = torch.zeros((self.len_idx,self.H,self.W)).float()
        self.image_depth = torch.zeros((self.len_idx,self.H,self.W)).float()

        self.demo = torch.zeros((1)).int()
        self.is_mapping_process_started = torch.zeros((1)).int()
        self.iter_shared = torch.zeros((1)).int()
        
        self.shared_cam.share_memory()
        self.shared_new_points.share_memory()
        self.shared_new_gaussians.share_memory()
        self.shared_target_gaussians.share_memory()
        self.end_of_dataset.share_memory_()
        self.is_tracking_keyframe_shared.share_memory_()
        self.is_mapping_keyframe_shared.share_memory_()
        self.target_gaussians_ready.share_memory_()
        self.new_points_ready.share_memory_()
        self.final_T_GTs.share_memory_()
        self.final_T_opts.share_memory_()

        self.image_r.share_memory_()
        self.image_g.share_memory_()
        self.image_b.share_memory_()
        self.image_depth.share_memory_()

        self.demo.share_memory_()
        self.is_mapping_process_started.share_memory_()
        self.iter_shared.share_memory_()

        self.gaussians = GaussianModel(self.sh_degree)  # For Debug
        
        self.demo[0] = args.demo
        self.tracker = Tracker(self)
        self.mapper = Mapper(self)
        # self.sender = Sender(self)

    def tracking(self, rank):
        self.tracker.run()
    
    def mapping(self, rank):
        self.mapper.run()

    # def sending(self, rank):
    #     self.sender.run()

    def run(self):
        processes = []
        for rank in range(2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, )) 
            # elif rank == 2:
            #     p = mp.Process(target=self.sending, args=(rank, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def get_init_image(self):
        for i in range(self.topic_num):
            shm = self.shared_memories[i]
            if shm is None:
                raise RuntimeError(f"Shared memory for camera is not available.")

            # 공유 메모리 버퍼에서 직접 데이터 읽기
            buffer = shm.buf

            if i == 0:
                r = np.frombuffer(buffer[4:], dtype=np.uint8, count=self.H * self.W * 3)[2::3].reshape((self.H, self.W))
                g = np.frombuffer(buffer[4:], dtype=np.uint8, count=self.H * self.W * 3)[1::3].reshape((self.H, self.W))
                b = np.frombuffer(buffer[4:], dtype=np.uint8, count=self.H * self.W * 3)[0::3].reshape((self.H, self.W))
                # BGR 이미지 생성
                rgb_image = np.stack([b, g, r], axis=-1)  # OpenCV는 BGR 순서 사용

            elif i == 1:
                depth_image = np.frombuffer(buffer[4:], dtype=np.float32, count=self.H * self.W).reshape((self.H, self.W))

        return rgb_image, depth_image

    def run_viewer(self, lower_speed=True):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            if time.time()-self.last_t < 1/self.viewer_fps and lower_speed:
                break
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                    # net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render_depth"]
                    # net_image = torch.concat([net_image,net_image,net_image], dim=0)
                    # net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=7.0) * 50).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path) 
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def set_downsample_filter( self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        # For tensor indexing, we need tuple
        pick_idxs = ((a+b).flatten(),)
        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        
        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx # * z_values
        y_pre = (v-self.cy)/self.fy # * z_values
        
        return pick_idxs, x_pre, y_pre

    def downsample_and_make_pointcloud(self, depth_img, rgb_img):
        
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale
        filter = torch.where((z_values!=0)&(z_values<=self.depth_trunc))
        # print(z_values[filter].min())
        # Trackable gaussians (will be used in tracking)
        z_values = z_values
        x = self.x_pre * z_values
        y = self.y_pre * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors
        
        # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)
        
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()
    
    def get_image_dirs(self, images_folder):
        if self.camera_parameters[8] == "replica":
            images_folder = os.path.join(self.dataset_path, "images")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            image_name = image_files[0].split(".")[0]
            depth_image_name = f"depth{image_name[5:]}"
        elif self.camera_parameters[8] == "tum":
            rgb_folder = os.path.join(self.dataset_path, "rgb")
            depth_folder = os.path.join(self.dataset_path, "depth")
            image_files = os.listdir(rgb_folder)
            depth_files = os.listdir(depth_folder)
 
        return image_files, depth_files

if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_path / output_path / verbose")
    parser.add_argument("--dataset_path", help="dataset path", default="assets")
    parser.add_argument("--config", help="caminfo", default="assets/config.txt")
    parser.add_argument("--output_path", help="output path", default="output/room0")
    parser.add_argument("--keyframe_th", default=0.5) #이전 프레임과 **얼마나 많이 겹치는가(혹은 매칭 정도가 충분한가)**
    parser.add_argument("--knn_maxd", default=99999.0) # kd tree, GICP 후보 탐색시 탐색반경 : 충분히 커서 거리제한 (x)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--demo", action='store_true', default=False)
    parser.add_argument("--overlapped_th", default=5e-4) # 프레임간 겹침판단할때 사용
    parser.add_argument("--max_correspondence_distance", default=0.02)
    parser.add_argument("--trackable_opacity_th", default=0.05)
    parser.add_argument("--overlapped_th2", default=5e-5)
    parser.add_argument("--save_results", action='store_true', default=None)

    args = parser.parse_args()

    gs_icp_slam = GS_ICP_SLAM(args) # CLI를 통해 argument 
    # gs_icp_slam.SLAM(1)
    gs_icp_slam.run()