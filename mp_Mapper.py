import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import copy
import random
import sys
import cv2
import numpy as np
import time
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from utils.loss_utils import l1_loss, ssim
from scene import GaussianModel
from gaussian_renderer import render, render_3, network_gui
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import open3d as o3d
import matplotlib.pyplot as plt
from mqtt_utils import connect_mqtt, save_json_to_file, save_binary_to_file, serialize_gaussian_to_json, send_to_mqtt, MQTT_TOPIC, serialize_gaussian_to_binary_all, serialize_gaussian_to_binary, sendTest_to_mqtt


class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug
        
class Mapper(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = float(slam.keyframe_th)
        self.trackable_opacity_th = slam.trackable_opacity_th
        self.save_results = slam.save_results
        self.iter_shared = slam.iter_shared

        self.camera_parameters = slam.camera_parameters
        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        self.downsample_size = slam.downsample_size
        self.topic_num = slam.topic_num
        self.start_idx = slam.start_idx
        self.end_idx = slam.end_idx
        self.len_idx = slam.len_idx
        
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0.,0.,1]])
        
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        
        # Camera poses
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [self.trajmanager.gt_poses[0]]
        # Keyframes(added to map gaussians)
        self.keyframe_idxs = []
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        self.start_trigger = False
        self.if_mapping_keyframe = False
        self.cam_t = []
        self.cam_R = []
        self.points_cat = []
        self.colors_cat = []
        self.rots_cat = []
        self.scales_cat = []
        self.trackable_mask = []
        self.from_last_tracking_keyframe = 0
        self.from_last_mapping_keyframe = 0
        self.scene_extent = 2.5
        if self.trajmanager.which_dataset == "replica":
            self.prune_th = 2.5
        else:
            self.prune_th = 10.0
        
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_size)

        # self.gaussians = GaussianModel(self.sh_degree)

        self.viewer = slam.viewer

        self.gaussians = slam.gaussians

        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0
        self.mapping_cams = []
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []
        
        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.final_T_GTs = slam.final_T_GTs
        self.final_T_opts = slam.final_T_opts

        self.image_r = slam.image_r
        self.image_g = slam.image_g
        self.image_b = slam.image_b
        self.image_depth = slam.image_depth

        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started
    
    def run(self):
        self.mapping()
        # self.mapping_test()
    
    def mapping(self):
        t = torch.zeros((1,1)).float().cuda()
        if self.verbose:
            network_gui.init("127.0.0.1", 6009)
        
        if self.viewer:
            rr.init("3dgsviewer")
            rr.connect()
        
        # Mapping Process is ready to receive first frame
        self.is_mapping_process_started[0] = 1
        
        # Wait for initial gaussians
        while not self.is_tracking_keyframe_shared[0]:
            time.sleep(1e-15)
            
        self.total_start_time_viewer = time.time()
        
        points, colors, rots, scales, z_values, trackable_filter = self.shared_new_gaussians.get_values()

        print("points: ")
        print(points.shape[0])
        
        self.gaussians.create_from_pcd2_tensor_LYS(points, colors, rots, scales, z_values, trackable_filter)
        self.gaussians.spatial_lr_scale = self.scene_extent
        self.gaussians.training_setup(self)
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree
        self.is_tracking_keyframe_shared[0] = 0
        self.global_index_send = 0
        
        if self.demo[0]:
            a = time.time()
            while (time.time()-a)<30.:
                print(30.-(time.time()-a))
                self.run_viewer()
        self.demo[0] = 0
        
        newcam = copy.deepcopy(self.shared_cam)
        newcam.on_cuda()

        self.mapping_cams.append(newcam)
        self.keyframe_idxs.append(newcam.cam_idx[0])
        self.new_keyframes.append(len(self.mapping_cams)-1)

        new_keyframe = False


        connect_mqtt() #LYS

        while True:         
            # print("Mapper: ")
            # print(len(self.gaussians.get_xyz))
            # print("SenderinM: ")
            # print(len(sender_gaussians.get_xyz))

            if self.end_of_dataset[0]:
                break
 
            if self.verbose:
                self.run_viewer()       
            
            if self.is_tracking_keyframe_shared[0]:
                # get shared gaussians
                points, colors, rots, scales, z_values, trackable_filter = self.shared_new_gaussians.get_values()
                
                old_count = self.gaussians.get_xyz.shape[0]                            
                

                # # Add new gaussians to map gaussians
                # points = torch.tensor([[0], [0], [0]], device='cuda')
                # colors = torch.tensor([[0.5]], device='cuda')
                # z_values = torch.tensor([[2], [1], [1]], device='cuda')
                # rots = torch.tensor([[0], [0], [0], [1]], device='cuda')

                self.gaussians.add_from_pcd2_tensor_LYS(points, colors, rots, scales, z_values, trackable_filter)
                
                new_count = self.gaussians.get_xyz.shape[0] 



                # Allocate new target points to shared memory
                target_points, target_rots, target_scales  = self.gaussians.get_trackable_gaussians_tensor(self.trackable_opacity_th)                
                self.shared_target_gaussians.input_values(target_points, target_rots, target_scales)
                self.target_gaussians_ready[0] = 1

                # Add new keyframe
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
                self.is_tracking_keyframe_shared[0] = 0

            elif self.is_mapping_keyframe_shared[0]:
                # get shared gaussians
                points, colors, rots, scales, z_values, _ = self.shared_new_gaussians.get_values()
                
                old_count = self.gaussians.get_xyz.shape[0]                 
                # Add new gaussians to map gaussians
                self.gaussians.add_from_pcd2_tensor_LYS(points, colors, rots, scales, z_values, [])

                new_count = self.gaussians.get_xyz.shape[0]                            
                # Add new keyframe
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
                self.is_mapping_keyframe_shared[0] = 0
        
            if len(self.mapping_cams)>0:
                
                # For Debug

                # # train once on new keyframe, and random
                # i = 0
                # while i < 100:
                #     i = i+1
                #     print(i)

                if len(self.new_keyframes) > 0:
                    train_idx = self.new_keyframes.pop(0)
                    viewpoint_cam = self.mapping_cams[train_idx]
                    new_keyframe = True
                else:
                    train_idx = random.choice(range(len(self.mapping_cams)))
                    viewpoint_cam = self.mapping_cams[train_idx]
                
                if self.training_stage==0:
                    gt_image = viewpoint_cam.original_image.cuda()
                    gt_depth_image = viewpoint_cam.original_depth_image.cuda()
                elif self.training_stage==1:
                    gt_image = viewpoint_cam.rgb_level_1.cuda()
                    gt_depth_image = viewpoint_cam.depth_level_1.cuda()
                elif self.training_stage==2:
                    gt_image = viewpoint_cam.rgb_level_2.cuda()
                    gt_depth_image = viewpoint_cam.depth_level_2.cuda()
                
                self.training=True

                # #For Debug 2

                # self.gaussians._xyz = torch.tensor([[0], [0], [0]], device='cuda')
                # self.gaussians._opacity = torch.tensor([[0.5]], device='cuda')
                # self.gaussians._scaling = torch.tensor([[2], [1], [1]], device='cuda')
                # self.gaussians._rotation = torch.tensor([[0], [0], [0], [1]], device='cuda')
                # #For Debug 2

                render_pkg = render_3(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)

                depth_image = render_pkg["render_depth"]
                image = render_pkg["render"]
                viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                
                mask = (gt_depth_image>0.)
                mask = mask.detach()
                # color_mask = torch.tile(mask, (3,1,1))
                gt_image = gt_image * mask
                
                # Loss
                Ll1_map, Ll1 = l1_loss(image, gt_image)
                L_ssim_map, L_ssim = ssim(image, gt_image)

                d_max = 10.
                Ll1_d_map, Ll1_d = l1_loss(depth_image/d_max, gt_depth_image/d_max)

                loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)
                loss_d = Ll1_d
                
                loss = loss_rgb + 0.1*loss_d

                
                #For Debug 5
                loss.backward()
                #For Debug 5


                with torch.no_grad():
                    if self.train_iter % 200 == 0:  # 200


                        # For Debug 4
                        removed_indices = self.gaussians.prune_large_and_transparent_LYS(0.005, self.prune_th)
                        # For Debug 4


                        print("가우시안 포인트 수: ", len(self.gaussians.get_xyz))
                        # print(f"Data shape: {self.gaussians.get_xyz.shape}")
                        

                        # LYS : 가우시안 전송
                        # copy_gaussian(self.gaussians)
                        # send_gaussian(self.global_index_send , gaussians=self.gaussians, min_optimization=100, binary_file="output/gaussians.flex")
                        send_gaussian_all(gaussians=self.gaussians, min_optimization=100, binary_file="output/gaussians.flex")
                        self.global_index_send  = self.global_index_send  + 1

                        # # LYS 예시: 추적하려는 ID 설정
                        # find_and_print_gaussian_by_id(gaussians=self.gaussians, target_id=2400)

                    
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)

                    # LYS: 현재 존재하는 모든 가우시안의 최적화 카운트를 +1
                    self.gaussians.optimize_count_plus()                    


                    # For Debug 1

                    # if new_keyframe and self.viewer:
                    #     current_i = copy.deepcopy(self.iter_shared[0])
                    #     rgb_np = image.cpu().numpy().transpose(1,2,0)
                    #     rgb_np = np.clip(rgb_np, 0., 1.0) * 255
                    #     # rr.set_time_sequence("step", current_i)
                    #     rr.set_time_seconds("log_time", time.time() - self.total_start_time_viewer)
                    #     rr.log("rendered_rgb", rr.Image(rgb_np))
                    #     new_keyframe = False

                    if new_keyframe and self.viewer:
                        # current_i = copy.deepcopy(self.iter_shared[0])
                        rgb_np = image.cpu().numpy().transpose(1,2,0)
                        rgb_np = np.clip(rgb_np, 0., 1.0) * 255
                        # rr.set_time_sequence("step", current_i)
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time_viewer)
                        rr.log("rendered_rgb", rr.Image(rgb_np))
                        new_keyframe = False

                    # For Debug 1
                        

                self.training = False
                self.train_iter += 1
                
                # torch.cuda.empty_cache()

                # For Debug
        #  종료후 : 저장
        print(f"최종 가우시안 포인트 수: {self.gaussians.get_xyz.shape}")                

   
        # points, colors = get_all_points_colors(self.gaussians)
        # rr.log(
        #     "GS_map",
        #     rr.Points3D(points, colors=colors, radii=0.01),
        #     timeless=True # 맨마지막꺼만 나오도록

        if self.verbose:
            while True:
                self.run_viewer(False)
        
        # End of data
        if self.save_results and not self.viewer:
            self.gaussians.save_ply(os.path.join(self.output_path, "scene.ply"))
        
        self.calc_2d_metric()


#################################################################################
################################### Functions ################################### 
#################################################################################


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
    

    def get_image_dirs(self, images_folder):
        color_paths = []
        depth_paths = []
        if self.trajmanager.which_dataset == "replica":
            images_folder = os.path.join(images_folder, "images")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            for key in tqdm(image_files):
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"    
                color_paths.append(f"{self.dataset_path}/images/{image_name}.jpg")            
                depth_paths.append(f"{self.dataset_path}/depth_images/{depth_image_name}.png")
                
            return color_paths, depth_paths
        elif self.trajmanager.which_dataset == "tum":
            return self.trajmanager.color_paths, self.trajmanager.depth_paths


    def calc_2d_metric(self):
        psnrs = []
        ssims = []
        lpips = []
        
        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")

        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        error_count = 0
        
        with torch.no_grad():
            for i in range(self.len_idx):
                print ("error: current step: ", i, "whole: ", (self.len_idx-1))
                gt_rgb = []
                gt_rgb_ = []
                gt_depth_ = []
                cam = self.mapping_cams[0]
                w2c = self.final_T_GTs[i]

                gt_rgb = torch.stack([self.image_b[i, :, :], self.image_g[i, :, :], self.image_r[i, :, :]])  # Shape: (3, H, W)
                gt_rgb_ = gt_rgb.float().cuda()/255
                gt_depth_ = self.image_depth[i].float().cuda().unsqueeze(0)  # Shape: (1, H, W)

                if w2c.norm(p='fro') < 1e-6:
                    print("skip sequence: ", i)
                    continue
                error_count += 1

                # rendered
                # cam class에서 T_wc = [R_wc, t_cw; 0, 1] 이라는 MZ한 형식을 사용                
                cam.R = w2c[:3,:3]
                cam.t = -np.matmul(cam.R.cpu().numpy().T, w2c[:3, 3])
                # cam.t = -torch.matmul(cam.R.T, torch.tensor(w2c[:3, 3], dtype=torch.float32, device="cuda"))
                # cam.t = -np.matmul(cam.R.transpose(), w2c[:3,3])
                # cam.t = w2c[:3,3]

                cam.image_width = gt_rgb_.shape[2]
                cam.image_height = gt_rgb_.shape[1]

                cam.update_matrix()

                # rendered rgb
                ours_rgb_ = render(cam, self.gaussians, self.pipe, self.background)["render"]
                ours_rgb_ = torch.clamp(ours_rgb_, 0., 1.).cuda()
                
                valid_depth_mask_ = (gt_depth_>0)
                
                gt_rgb_ = gt_rgb_ * valid_depth_mask_
                ours_rgb_ = ours_rgb_ * valid_depth_mask_

                # rr rendering
                if self.viewer:
                    gt_np = gt_rgb_.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255  # Convert to [H, W, 3]
                    ours_np = ours_rgb_.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255  # Convert to [H, W, 3]
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time_viewer)
                    rr.log("cam/current", rr.Image(gt_np))
                    rr.log("rendered_rgb", rr.Image(ours_np))
                    time.sleep(1e-1)
                
                square_error = (gt_rgb_-ours_rgb_)**2
                mse_error = torch.mean(torch.mean(square_error, axis=2))
                psnr = mse2psnr(mse_error)
                psnrs += [psnr.detach().cpu()]
                _, ssim_error = ssim(ours_rgb_, gt_rgb_)
                ssims += [ssim_error.detach().cpu()]
                lpips_value = cal_lpips(gt_rgb_.unsqueeze(0), ours_rgb_.unsqueeze(0))
                lpips += [lpips_value.detach().cpu()]
                
                # if self.save_results and ((i+1)%100==0 or i==len(image_names)-1):
                #     ours_rgb = np.asarray(ours_rgb_.detach().cpu()).squeeze().transpose((1,2,0))
                    
                #     axs[0].set_title("gt rgb")
                #     axs[0].imshow(gt_rgb)
                #     axs[0].axis("off")
                #     axs[1].set_title("rendered rgb")
                #     axs[1].imshow(ours_rgb)
                #     axs[1].axis("off")
                #     plt.suptitle(f'{i+1} frame')
                #     plt.pause(1e-15)
                #     plt.savefig(f"{self.output_path}/result_{i}.png")
                #     plt.cla()
                
                torch.cuda.empty_cache()
            
            psnrs = np.array(psnrs)
            ssims = np.array(ssims)
            lpips = np.array(lpips)
            
            print(f"PSNR: {psnrs.mean():.2f}\
                  \nSSIM: {ssims.mean():.3f}\
                  \nLPIPS: {lpips.mean():.3f}")
            


def mse2psnr(x):
    return -10.*torch.log(x)/torch.log(torch.tensor(10.))


#LYS
def get_all_points_colors(gaussians):
    """
    가우시안 모델에서 모든 포인트와 색상을 가져옵니다.
    Args:
        gaussians (GaussianModel): 가우시안 모델 객체
    Returns:
        tuple: (points Nx3, colors Nx3) 형태의 NumPy 배열
    """
    if gaussians.get_xyz.shape[0] == 0:
        return None, None

    # 좌표를 NumPy 배열로 변환
    xyz_np = gaussians.get_xyz.detach().cpu().numpy()

    # DC 성분에서 색상 추출 (f_dc 채널 사용)
    f_dc = gaussians._features_dc.detach().cpu().numpy()  # (N, 1, 3)
    f_dc_squeezed = f_dc.squeeze(1)
    colors_np = f_dc_squeezed  
    
    f_rest = gaussians._features_rest.detach().cpu().numpy()  # (N, 0?, 3)
    scale_np = gaussians._scaling.detach().cpu().numpy() # (N,3)
    quat_np = gaussians._rotation.detach().cpu().numpy() # (N,4)
    opacity_np = gaussians._opacity.detach().cpu().numpy() # (N,1)
    # self.xyz_gradient_accum = torch.empty(0)
    # self.denom = torch.empty(0)

    return xyz_np, colors_np, opacity_np, scale_np, quat_np


def send_gaussian(global_index_send, gaussians, min_optimization=0, binary_file="output/gaussians.flex"):
    """
    전송된 적 없고 최적화 횟수가 min_optimization 이상인 가우시안을 전송하거나 저장.

    Args:
        gaussians (GaussianModel): GaussianModel 객체
        min_optimization (int): 전송 기준 최적화 횟수 (기본값 200)
        json_file (str): 저장할 JSON 파일 경로
        binary_file (str): 저장할 바이너리 파일 경로
    """

    # 누적 전송 포인트 수를 기록할 변수
    if not hasattr(send_gaussian, "total_sent_points"):
        send_gaussian.total_sent_points = 0

    # 전송되지 않은 가우시안 인덱스 가져오기
    unsent_indices = gaussians.get_unsent_ids(min_optimization=min_optimization)

    if unsent_indices.size > 0:
        # 필요한 데이터 추출
        xyz_np, colors_np, opacity_np, scale_np, quat_np = get_all_points_colors(gaussians)

        unsent_xyz = xyz_np[unsent_indices]
        unsent_colors = colors_np[unsent_indices]
        unsent_opacity = opacity_np[unsent_indices]
        unsent_colors_rgba = np.concatenate([unsent_colors, unsent_opacity], axis=-1)
        unsent_scales = scale_np[unsent_indices]
        unsent_rots = quat_np[unsent_indices]
        unsent_ids = gaussians.gaussian_ids[unsent_indices, 0].astype(int)  # ID 값 추출 (정수형 변환)
   

        # 바이너리 직렬화
        serialized_data2 = serialize_gaussian_to_binary(
            global_index_send,
            new_xyz=unsent_xyz,
            new_colors_rgba=unsent_colors_rgba,
            new_scales=unsent_scales,
            new_rots=unsent_rots,
            new_ids=unsent_ids
        )
        
        save_binary_to_file(binary_file, serialized_data2)        
        send_to_mqtt(MQTT_TOPIC, serialized_data2)

        # 전송된 포인트의 현재 최적화 횟수를 sent 열에 기록
        gaussians.update_sent_ids(unsent_indices)

        # 누적 전송 포인트 수 업데이트
        send_gaussian.total_sent_points += unsent_indices.size

        # JSON 직렬화
        # serialized_data = serialize_gaussian_to_json(
        #     new_xyz=unsent_xyz,
        #     new_colors_rgba=unsent_colors_rgba,
        #     new_scales=unsent_scales,
        #     new_rots=unsent_rots,
        #     new_ids=unsent_ids
        # )
        # 파일로 저장
        #save_json_to_file(json_file, serialized_data)

        print(f"전송 포인트 {unsent_indices.size}, 누적 전송 포인트 {send_gaussian.total_sent_points}.")
    else:
        print(f"최적화 횟수가 {min_optimization} 이상인 포인트가 없습니다.")



def send_gaussian_all_frag(gaussians, min_optimization=200, binary_file="output/gaussians.flex", batch_size=1000):
    """
    전송된 적 없고 최적화 횟수가 min_optimization 이상인 가우시안을 batch_size 단위로 전송하거나 저장.

    Args:
        gaussians (GaussianModel): GaussianModel 객체
        min_optimization (int): 전송 기준 최적화 횟수 (기본값 200)
        binary_file (str): 저장할 바이너리 파일 경로
        batch_size (int): 한 번에 전송할 가우시안 개수 (기본값 1000)
    """

    if not hasattr(send_gaussian, "total_sent_points"):
        send_gaussian.total_sent_points = 0

    minopt_indices = gaussians.get_min_opt_ids(min_optimization=min_optimization)

    if minopt_indices.size > 0:
        xyz_np, colors_np, opacity_np, scale_np, quat_np = get_all_points_colors(gaussians)

        minopt_xyz = xyz_np[minopt_indices]
        minopt_colors = colors_np[minopt_indices]
        minopt_opacity = opacity_np[minopt_indices]
        minopt_colors_rgba = np.concatenate([minopt_colors, minopt_opacity], axis=-1)
        minopt_scales = scale_np[minopt_indices]
        minopt_rots = quat_np[minopt_indices]
        minopt_ids = gaussians.gaussian_ids[minopt_indices, 0].astype(int)

        total_batches = (minopt_indices.size + batch_size - 1) // batch_size  # 전체 배치 수

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, minopt_indices.size)

            batch_xyz = minopt_xyz[start_idx:end_idx]
            batch_colors_rgba = minopt_colors_rgba[start_idx:end_idx]
            batch_scales = minopt_scales[start_idx:end_idx]
            batch_rots = minopt_rots[start_idx:end_idx]
            batch_ids = minopt_ids[start_idx:end_idx]

            # 바이너리 직렬화
            serialized_data = serialize_gaussian_to_binary(
                new_xyz=batch_xyz,
                new_colors_rgba=batch_colors_rgba,
                new_scales=batch_scales,
                new_rots=batch_rots,
                new_ids=batch_ids
            )

            # 헤더 생성 (전체 조각 수와 현재 조각 번호 추가)
            header = np.array([total_batches, batch_num + 1], dtype=np.uint32).tobytes()
            full_data = header + serialized_data

            # 바이너리 파일 저장 및 MQTT 전송
            save_binary_to_file(binary_file, full_data)
            send_to_mqtt(MQTT_TOPIC, full_data)

            print(f"Batch {batch_num + 1}/{total_batches} 전송 완료")

        # 전송된 포인트의 현재 최적화 횟수를 sent 열에 기록
        gaussians.update_sent_ids(minopt_indices)

        print(f"총 {minopt_indices.size} 포인트를 {total_batches} 배치로 전송 완료")
    else:
        print(f"최적화 횟수가 {min_optimization} 이상인 포인트가 없습니다.")


def send_gaussian_all(gaussians, min_optimization=200, binary_file="output/gaussians.flex"):
    """
    전송된 적 없고 최적화 횟수가 min_optimization 이상인 가우시안을 전송하거나 저장.

    Args:
        gaussians (GaussianModel): GaussianModel 객체
        min_optimization (int): 전송 기준 최적화 횟수 (기본값 200)
        json_file (str): 저장할 JSON 파일 경로
        binary_file (str): 저장할 바이너리 파일 경로
    """

    # 누적 전송 포인트 수를 기록할 변수
    if not hasattr(send_gaussian, "total_sent_points"):
        send_gaussian.total_sent_points = 0

    # 전송되지 않은 가우시안 인덱스 가져오기
    # unsent_indices = gaussians.get_unsent_ids(min_optimization=min_optimization)
    minopt_indices = gaussians.get_min_opt_ids(min_optimization=min_optimization)

    if minopt_indices.size > 0:
        # 필요한 데이터 추출
        xyz_np, colors_np, opacity_np, scale_np, quat_np = get_all_points_colors(gaussians)

        minopt_xyz = xyz_np[minopt_indices]
        minopt_colors = colors_np[minopt_indices]
        minopt_opacity = opacity_np[minopt_indices]
        minopt_colors_rgba = np.concatenate([minopt_colors, minopt_opacity], axis=-1)
        minopt_scales = scale_np[minopt_indices]
        minopt_rots = quat_np[minopt_indices]
        minopt_ids = gaussians.gaussian_ids[minopt_indices, 0].astype(int)  # ID 값 추출 (정수형 변환)
   

        # 바이너리 직렬화
        serialized_data2 = serialize_gaussian_to_binary_all(
            new_xyz=minopt_xyz,
            new_colors_rgba=minopt_colors_rgba,
            new_scales=minopt_scales,
            new_rots=minopt_rots,
            new_ids=minopt_ids
        )
        
        save_binary_to_file(binary_file, serialized_data2)        
        send_to_mqtt(MQTT_TOPIC, serialized_data2)

        # 전송된 포인트의 현재 최적화 횟수를 sent 열에 기록
        gaussians.update_sent_ids(minopt_indices)

        # 누적 전송 포인트 수 업데이트
        # send_gaussian.total_sent_points += minopt_indices.size

        # JSON 직렬화
        # serialized_data = serialize_gaussian_to_json(
        #     new_xyz=unsent_xyz,
        #     new_colors_rgba=unsent_colors_rgba,
        #     new_scales=unsent_scales,
        #     new_rots=unsent_rots,
        #     new_ids=unsent_ids
        # )
        # 파일로 저장
        #save_json_to_file(json_file, serialized_data)

        print(f"전송 포인트 {minopt_indices.size}")
    # else:
    #     print(f"최적화 횟수가 {min_optimization} 이상인 포인트가 없습니다.")

def find_and_print_gaussian_by_id(gaussians, target_id):
    """
    특정 ID의 가우시안 데이터를 찾고 정보를 출력합니다.

    Args:
        gaussians (GaussianModel): GaussianModel 객체
        target_id (int): 찾고자 하는 가우시안의 ID
    """
    # 현재 gaussian_ids에서 target_id의 위치를 찾음
    target_index = np.searchsorted(gaussians.gaussian_ids[:, 0], target_id)

    # ID가 실제로 존재하는지 확인
    if target_index < len(gaussians.gaussian_ids) and gaussians.gaussian_ids[target_index, 0] == target_id:
        # target_index를 사용해 데이터 추출
        target_point = gaussians.get_xyz[target_index].cpu().numpy()  # 위치 좌표
        target_color = gaussians.get_features[target_index, :3, 0].cpu().numpy()  # 색상
        target_opacity = gaussians.get_opacity[target_index].cpu().numpy()  # Opacity
        print(f"ID {target_id} - Point: {target_point}, Color: {target_color}, Opacity: {target_opacity}")
    else:
        print(f"ID {target_id}는 현재 가우시안에 없습니다.")



def display_pc(points1, points2):
  
    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(points1)

    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(points2)

    pc1.paint_uniform_color([1, 0, 0])  # 빨간색
    pc2.paint_uniform_color([0, 1, 0])  # 초록색

    o3d.visualization.draw_geometries([pc1, pc2], window_name="3D Point Clouds")




