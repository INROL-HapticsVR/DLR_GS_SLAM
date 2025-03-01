import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
from random import randint
import sys
import cv2
import numpy as np
import open3d as o3d
import pygicp
import time
from scipy.spatial.transform import Rotation
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from gaussian_renderer import render, render_2, network_gui
from tqdm import tqdm
from multiprocessing import shared_memory

from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

import scipy.linalg
from scipy.optimize import minimize

class Tracker(SLAMParameters):
    def __init__(self, slam):
         # gs_icp_slam.py에서 Tacker(self)로 생성 
        # --> 즉 GS_ICP_SLAM()객체 = self  = 이 파일에서 slam
      
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = slam.keyframe_th
        self.knn_max_distance = slam.knn_max_distance
        self.overlapped_th = slam.overlapped_th
        self.overlapped_th2 = slam.overlapped_th2
        self.test = slam.test
        self.rerun_viewer = slam.rerun_viewer
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
        self.max_fps = slam.max_fps
        self.image_num = slam.image_num
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0.,0.,1]])
        
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        self.max_correspondence_distance = slam.max_correspondence_distance
        self.reg = pygicp.FastGICP()
        
        # Camera poses
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [self.trajmanager.gt_poses[0]]
        # Keyframes(added to map gaussians)
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        
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
        
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_size)

        # Share
        self.train_iter = 0
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
        self.new_points_ready = slam.new_points_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started
    
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
        
    def run(self):
        self.tracking()

    def tracking(self):
        tt = torch.zeros((1,1)).float().cuda()
        
        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect()
        
        self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
        self.reg.set_max_knn_distance(self.knn_max_distance)
        if_mapping_keyframe = False

        self.total_start_time = time.time()

        iter_count = 0
        points_total = np.empty((0, 3))
        points_prev = np.empty((0, 3))
        pose_opt = []
        pose_prop_prev = []
        # for iter_count in range(self.num_images):
        
        if self.image_num < 1:
            unlimited_Length = True
        else :
            unlimited_Length = False
            pbar = tqdm(total=self.image_num)

        while iter_count < self.image_num or unlimited_Length:
            self.iter_shared[0] = iter_count

            rgb_index, depth_index, pose_index, current_image, depth_image, R_wc, t_wc, T_depth = self.get_sharedmemory()

            # Make pointcloud
            points, colors, z_values, trackable_filter = self.downsample_and_make_pointcloud2(depth_image, current_image)           
            
            if self.iteration_images == 0:
                current_pose = self.poses[-1]

                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose)
                
                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)

                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(
                        "cam/current",
                        # rr.Transform3D(translation=self.poses[-1][:3,3],
                        #             rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[-1][:3,:3])).as_quat()))
                        rr.Transform3D(translation=t_wc,
                        rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(R_wc)).as_quat()))
                    )
                    rr.log(
                        "cam/current",
                        rr.Pinhole(
                            resolution=[self.W, self.H],
                            image_from_camera=self.cam_intrinsic,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        )
                    )
                    rr.log(
                        "cam/current",
                        rr.Image(current_image)
                    )

                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()
                
                # transform current points
                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                self.reg.set_input_target(points)

                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]            

                # For Debug 3
                self.reg.set_target_filter(num_trackable_points, input_filter)
                self.reg.calculate_target_covariance_with_filter()

                rots = self.reg.get_target_rotationsq()
                scales = self.reg.get_target_scales()
                rots = np.reshape(rots, (-1,4))
                scales = np.reshape(scales, (-1,3))
                
                # Assign first gaussian to shared memory
                self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                       torch.tensor(rots), torch.tensor(scales), 
                                                       torch.tensor(z_values), torch.tensor(trackable_filter))

                # Add first keyframe
                depth_image = depth_image.astype(np.float32)/self.depth_scale
                # self.shared_cam.setup_cam(R, T, current_image, depth_image)
                self.shared_cam.setup_cam(R_wc, T_depth, current_image, depth_image)
                self.shared_cam.cam_idx[0] = self.iteration_images
                
                self.is_tracking_keyframe_shared[0] = 1
                
                while self.demo[0]:
                    time.sleep(1e-15)
                    self.total_start_time = time.time()
                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.000004*self.W*self.downsample_size))
            else:
                # GICP
                # For Debug 1
                self.reg.set_input_source(points)
                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                self.reg.set_source_filter(num_trackable_points, input_filter)
                
                initial_pose = self.poses[-1]

                print("here???????????????????????????????????????????????????????")
                print("here???????????????????????????????????????????????????????")
                print("here???????????????????????????????????????????????????????")

                current_pose = self.reg.align(initial_pose)
                # current_pose = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)

                print("here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                
                self.poses.append(current_pose)
                # For Debug 1




                # For Debug 2
                pose_prop_curr = np.eye(4)  # 4x4 단위 행렬 생성
                pose_prop_curr[:3, :3] = R_wc  # 회전 행렬 할당
                pose_prop_curr[:3, 3] = t_wc   # 변환 벡터 할당
                # pose_opt_prev = pose_opt[-1]
                

                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()


                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                points_prev = points


                # print("corres: ")
                # print(corres_idx.shape[0])
                # print(corres_idx)
                # print("cov: ")
                # print(cov_s.shape[0])
                # print(cov_s)

                # For Debug 2


                # 포인트 정합
                # points_total, trackable_filter, points_new_len = self.update_total_points_parallel(points, points_total, 0.01)
                # points_prev, trackable_filter, points_new_len = self.update_total_points_prev(points, points_prev, 0.01)
                points_total, trackable_filter, points_new_len = self.update_total_points(points, points_total, 0.01)


                # total_points = self.update_total_points(points, total_points, 0.01)

                # print ("ddddd")
                # print (len(points_total))
                # print (points_new_len)

                # For Debug 포인트 정합
                
                # Use only trackable points when tracking
                # target_corres, distances = self.reg.get_source_correspondence() # get associated points source points

                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)

                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(
                        "cam/current",
                        # rr.Transform3D(translation=self.poses[-1][:3,3],
                        #             rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[-1][:3,:3])).as_quat()))
                        rr.Transform3D(translation = t_wc,
                        rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(R_wc)).as_quat()))
                    )
                    # translation=self.poses[-1][:3,3]
                    
                    rr.log(
                        "cam/current",
                        rr.Pinhole(
                            resolution=[self.W, self.H],
                            image_from_camera=self.cam_intrinsic,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        )
                    )
                    rr.log(
                        "cam/current",
                        rr.Image(current_image)
                        # rr.Image(depth_image)
                    )
                
                # Keyframe selection #
                # Tracking keyframe
                # len_corres = len(np.where(distances<self.overlapped_th)[0]) # 5e-4 self.overlapped_th

                if  (self.iteration_images >= self.image_num-1 \
                    or points_new_len/len(points) > self.keyframe_th):
                    if_tracking_keyframe = True
                    self.from_last_tracking_keyframe = 0                    
                else:
                    if_tracking_keyframe = False
                    self.from_last_tracking_keyframe += 1
                
                # Mapping keyframe
                if (self.from_last_tracking_keyframe) % self.keyframe_freq == 0:
                    if_mapping_keyframe = True
                else:
                    if_mapping_keyframe = False
                
                if if_tracking_keyframe:             
                    while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                        time.sleep(1e-15)
                    rots = np.array(self.reg.get_source_rotationsq())
                    rots = np.reshape(rots, (-1,4))

                    R_d = Rotation.from_matrix(R_wc)    # from camera R
                    R_d_q = R_d.as_quat()            # xyzw
                    rots = self.quaternion_multiply(R_d_q, rots)
                    
                    scales = np.array(self.reg.get_source_scales())
                    scales = np.reshape(scales, (-1,3))

                    points_new, colors_new, rots_new, scales_new, z_values_new \
                        = self.filter_new_values(points, colors, rots, scales, z_values, trackable_filter)
                    trackable_filter_new = np.arange(0, points_new_len, dtype=np.int64)


                    #For Debug 2
                    
                    # points_new, colors_new, rots_new, scales_new, z_values_new, trackable_filter_new = self.replace_points()

                    #For Debug 2


                    self.shared_new_gaussians.input_values(torch.tensor(points_new), torch.tensor(colors_new), 
                                                       torch.tensor(rots_new), torch.tensor(scales_new), 
                                                       torch.tensor(z_values_new), torch.tensor(trackable_filter_new))
                    
                    if self.rerun_viewer:
                        # rr.set_time_sequence("step", self.iteration_images)
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                        rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points_new, colors=colors_new, radii=0.000004*self.W*self.downsample_size))
                    
                    

                    # Add new keyframe
                    depth_image = depth_image.astype(np.float32)/self.depth_scale
                    self.shared_cam.setup_cam(R_wc, T_depth, current_image, depth_image)
                    self.shared_cam.cam_idx[0] = self.iteration_images
                    
                    self.is_tracking_keyframe_shared[0] = 1
                    
                    # Get new target point
                    while not self.target_gaussians_ready[0]:
                        time.sleep(1e-15)
                    target_points, target_rots, target_scales = self.shared_target_gaussians.get_values_np()

                    self.reg.set_input_target(target_points)
                    self.reg.set_target_covariances_fromqs(target_rots.flatten(), target_scales.flatten())
                    self.target_gaussians_ready[0] = 0

                elif if_mapping_keyframe:
                    
                    while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                        time.sleep(1e-15)
                    
                    rots = np.array(self.reg.get_source_rotationsq())
                    rots = np.reshape(rots, (-1,4))

                    R_d = Rotation.from_matrix(R_wc)    # from camera R
                    R_d_q = R_d.as_quat()            # xyzw
                    rots = self.quaternion_multiply(R_d_q, rots)
                                        
                    scales = np.array(self.reg.get_source_scales())
                    scales = np.reshape(scales, (-1,3))

                    points_new, colors_new, rots_new, scales_new, z_values_new \
                        = self.filter_new_values(points, colors, rots, scales, z_values, trackable_filter)
                    trackable_filter_new = np.arange(0, points_new_len, dtype=np.int64)


                    # For Debug 3

                    # points_new, colors_new, rots_new, scales_new, z_values_new, trackable_filter_new = self.replace_points()

                    # For Debug 3


                    self.shared_new_gaussians.input_values(torch.tensor(points_new), torch.tensor(colors_new), 
                                                       torch.tensor(rots_new), torch.tensor(scales_new), 
                                                       torch.tensor(z_values_new), torch.tensor(trackable_filter_new))
                    
                    # self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                    #                                    torch.tensor(rots), torch.tensor(scales), 
                    #                                    torch.tensor(z_values), torch.tensor(trackable_filter))
                    
                    
                    # Add new keyframe
                    depth_image = depth_image.astype(np.float32)/self.depth_scale
                    # self.shared_cam.setup_cam(R, T, current_image, depth_image)
                    self.shared_cam.setup_cam(R_wc, T_depth, current_image, depth_image)
                    self.shared_cam.cam_idx[0] = self.iteration_images
                    
                    self.is_mapping_keyframe_shared[0] = 1
            if unlimited_Length:
                print("iteration: " + str(iter_count))
            else :
                pbar.update(1)
            
            while 1/((time.time() - self.total_start_time)/(self.iteration_images+1)) > self.max_fps:    #30. float(self.test)
                time.sleep(1e-15)
                
            self.iteration_images += 1
            iter_count = iter_count+1
        
        # Tracking end
        if not unlimited_Length:
            pbar.close()
        self.final_pose[:,:,:] = torch.tensor(self.poses).float()
        self.end_of_dataset[0] = 1
        
        print(f"System FPS: {1/((time.time()-self.total_start_time)/iter_count):.2f}")
        print(f"ATE RMSE: {self.evaluate_ate(self.trajmanager.gt_poses, self.poses)*100.:.2f}")

    
    def get_images(self, images_folder):
        rgb_images = []
        depth_images = []
        if self.trajmanager.which_dataset == "replica":
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            for key in tqdm(image_files): 
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"
                
                rgb_image = cv2.imread(f"{self.dataset_path}/images/{image_name}.jpg")
                depth_image = np.array(o3d.io.read_image(f"{self.dataset_path}/depth_images/{depth_image_name}.png"))
                
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images
        elif self.trajmanager.which_dataset == "tum":
            for i in tqdm(range(len(self.trajmanager.color_paths))):
                rgb_image = cv2.imread(self.trajmanager.color_paths[i])
                depth_image = np.array(o3d.io.read_image(self.trajmanager.depth_paths[i]))
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images

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

    def quaternion_multiply(self, q1, Q2):
        # q1*Q2
        x0, y0, z0, w0 = q1
        
        return np.array([w0*Q2[:,0] + x0*Q2[:,3] + y0*Q2[:,2] - z0*Q2[:,1],
                        w0*Q2[:,1] + y0*Q2[:,3] + z0*Q2[:,0] - x0*Q2[:,2],
                        w0*Q2[:,2] + z0*Q2[:,3] + x0*Q2[:,1] - y0*Q2[:,0],
                        w0*Q2[:,3] - x0*Q2[:,0] - y0*Q2[:,1] - z0*Q2[:,2]]).T

    def set_downsample_filter( self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0

        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        pick_idxs = ((a+b).flatten(),)

        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
         
        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx # * z_values
        y_pre = (v-self.cy)/self.fy # * z_values
       
        return pick_idxs, x_pre, y_pre
    

    def set_downsample_filter_refine(self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0, int(self.H / sample_interval) + 1)
        h_val = h_val - 1
        h_val[0] = 0

        # Debug 1

        h_val = h_val * self.W
        a, b = torch.meshgrid(h_val, torch.arange(0, self.W, sample_interval))
        pick_idxs = ((a + b).flatten(),)  # For tensor indexing, we need tuple

        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0, self.H), torch.arange(0, self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]

        # Define regions
        region_height = self.H // 3
        region_width = self.W // 3

        # Identify the 8th region
        region8_mask = (
            (v >= 2 * region_height) & (v < self.H) &  # Bottom row
            (u >= region_width) & (u < 2 * region_width)  # Middle column
        )

        # Downsample scale for the 8th region
        fine_sample_interval = sample_interval // 2
        fine_h_val = fine_sample_interval * torch.arange(0, int(self.H / fine_sample_interval) + 1)
        fine_h_val = fine_h_val - 1
        fine_h_val[0] = 0

        fine_h_val = fine_h_val * self.W
        fine_a, fine_b = torch.meshgrid(fine_h_val, torch.arange(0, self.W, fine_sample_interval))
        fine_pick_idxs = ((fine_a + fine_b).flatten(),)

        fine_v, fine_u = torch.meshgrid(torch.arange(0, self.H), torch.arange(0, self.W))
        fine_u = fine_u.flatten()[fine_pick_idxs]
        fine_v = fine_v.flatten()[fine_pick_idxs]

        # Identify the 8th region
        fine_region8_mask = (
            (fine_v >= 2 * region_height) & (fine_v < self.H) &  # Bottom row
            (fine_u >= region_width) & (fine_u < 2 * region_width)  # Middle column
        )


        # Combine pick_idxs and fine_pick_idxs into a single index
        combined_idxs = []
        for i, idx in enumerate(pick_idxs[0]):
            if region8_mask[i]:  # If this index is in the 8th region
                combined_idxs.append(fine_pick_idxs[0][fine_region8_mask])  # Use fine indices
            else:
                combined_idxs.append(idx)  # Use coarse indices

        combined_idxs = torch.cat(combined_idxs)


        # Combine normal and fine sampling
        u_combined = torch.cat([u[~region8_mask], fine_u[fine_region8_mask]])
        v_combined = torch.cat([v[~region8_mask], fine_v[fine_region8_mask]])

        # Calculate xy values, not multiplied with z_values
        x_pre = (u_combined - self.cx) / self.fx  # * z_values
        y_pre = (v_combined - self.cy) / self.fy  # * z_values

        return (combined_idxs,), x_pre, y_pre


    def downsample_and_make_pointcloud2(self, depth_img, rgb_img):
        
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale
        zero_filter = torch.where(z_values!=0)
        filter = torch.where(z_values[zero_filter]<=self.depth_trunc)
        # print(z_values[filter].min())
        # Trackable gaussians (will be used in tracking)
        z_values = z_values[zero_filter]
        x = self.x_pre[zero_filter] * z_values
        y = self.y_pre[zero_filter] * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors[zero_filter]
        
        # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)
        
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()
    
    def eliminate_overlapped2(self, distances, threshold):
        
        # plt.hist(distances, bins=np.arange(0.,0.003,0.00001))
        # plt.show()
        new_p_indices = np.where(distances>threshold)    # 5e-5
        
        return new_p_indices
        
    def align(self, model, data):

        np.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1).reshape((3,-1))
        data_zerocentered = data - data.mean(1).reshape((3,-1))

        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
            S[2, 2] = -1
        rot = U*S*Vh
        trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

        model_aligned = rot * model + trans
        alignment_error = model_aligned - data

        trans_error = np.sqrt(np.sum(np.multiply(
            alignment_error, alignment_error), 0)).A[0]

        return rot, trans, trans_error

    def evaluate_ate(self, gt_traj, est_traj):

        gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
        gt_traj_pts_arr = np.array(gt_traj_pts)
        gt_traj_pts_tensor = torch.tensor(gt_traj_pts_arr)
        gt_traj_pts = torch.stack(tuple(gt_traj_pts_tensor)).detach().cpu().numpy().T

        est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]
        est_traj_pts_arr = np.array(est_traj_pts)
        est_traj_pts_tensor = torch.tensor(est_traj_pts_arr)
        est_traj_pts = torch.stack(tuple(est_traj_pts_tensor)).detach().cpu().numpy().T

        _, _, trans_error = self.align(gt_traj_pts, est_traj_pts)

        avg_trans_error = trans_error.mean()

        return avg_trans_error
    
    def Q2R(self, quaternion):
        # Normalize the quaternion
        quaternion = quaternion / np.linalg.norm(quaternion)

        # Extract individual components of the quaternion
        q0, q1, q2, q3 = quaternion

        # Compute the rotation matrix
        Rotation_m = np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)],
            [2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0**2 - q1**2 - q2**2 + q3**2]
        ])
        return Rotation_m


    def get_sharedmemory(self):
        # rgb_image, depth_image, R, T = None, None, None, None
        # rgb_index, depth_index, pose_index = -1, -1, -1

        while True:
            # time.sleep(SHM_DELAY)
            for i in range(self.topic_num):
                shm = self.shared_memories[i]
                if shm is None:
                    raise RuntimeError(f"Shared memory for camera is not available.")

                # Read directly from the shared memory buffer
                buffer = shm.buf

                if i == 0:
                    rgb_locker = np.frombuffer(buffer[0:1], dtype=np.uint8)[0]
                    rgb_index = np.frombuffer(buffer[1:5], dtype=np.uint32)[0]
                    r = np.frombuffer(buffer[5:], dtype=np.uint8, count=self.H * self.W * 3)[2::3].reshape((self.H, self.W))
                    g = np.frombuffer(buffer[5:], dtype=np.uint8, count=self.H * self.W * 3)[1::3].reshape((self.H, self.W))
                    b = np.frombuffer(buffer[5:], dtype=np.uint8, count=self.H * self.W * 3)[0::3].reshape((self.H, self.W))

                    # Create BGR image
                    rgb_image = np.stack([b, g, r], axis=-1)  # OpenCV uses BGR order

                elif i == 1:
                    depth_locker = np.frombuffer(buffer[0:1], dtype=np.uint8)[0]
                    depth_index = np.frombuffer(buffer[1:5], dtype=np.uint32)[0]
                    depth_image = np.frombuffer(buffer[5:], dtype=np.float32, count=self.H * self.W).reshape((self.H, self.W))

                elif i == 2:
                    pose_locker = np.frombuffer(buffer[0:1], dtype=np.uint8)[0]
                    pose_index = np.frombuffer(buffer[1:5], dtype=np.uint32)[0]
                    tmp = np.frombuffer(buffer[5:], dtype=np.float32, count=7)
                    Quater = tmp[3:7]
                    R_uc = self.Q2R(Quater)
                    R_wu = np.array([
                        [0, -1, 0],
                        [-1, 0, 0],
                        [0, 0, -1]
                    ])
                    R_wc = R_wu @ R_uc
                    t_wc = R_wu @ tmp[0:3]
                    T_depth = -np.matmul(R_wc.transpose(), t_wc)


            # Check if all indices are synchronized
            # if rgb_index > 275:
            #     while 1: time.sleep(1e-1) 

            if rgb_index == depth_index == pose_index and rgb_locker == depth_locker == pose_locker == 1:
            # if rgb_index == depth_index == pose_index:
                break

        # Return global index and synchronized data
        return rgb_index, depth_index, pose_index, rgb_image, depth_image, R_wc, t_wc, T_depth


    def cleanup(self):
        # Shared memory 해제
        for shm in self.shared_memories:
            if shm is not None:
                shm.close()

    def filter_new_values(self, points, colors, rots, scales, z_values, trackable_filter):
        # Filter values where trackable_filter is 1
        mask = trackable_filter == 1
        points_new = points[mask]
        colors_new = colors[mask]
        rots_new = rots[mask]
        scales_new = scales[mask]
        z_values_new = z_values[mask]

        return points_new, colors_new, rots_new, scales_new, z_values_new
    

    def update_total_points(self, points, total_points, threshold=0.01):
        if total_points.size == 0:
            # 모든 점이 새로운 점으로 간주되므로 trackable_filter는 모두 1
            trackable_filter = np.ones(points.shape[0], dtype=int)
            return points, trackable_filter, points.shape[0]

        # Combine new points with total_points
        combined_points = np.vstack((total_points, points))

        # Create KD-Tree for combined points
        tree = cKDTree(combined_points)

        # Find unique points
        unique_indices = []
        visited = set()
        trackable_filter = np.zeros(combined_points.shape[0], dtype=int)  # Initialize filter

        for i in range(combined_points.shape[0]):
            if i in visited:
                continue
            # Query points within the threshold
            neighbors = tree.query_ball_point(combined_points[i], threshold)
            visited.update(neighbors)  # Mark neighbors as visited
            unique_indices.append(i)  # Keep the first occurrence

        # Extract unique points
        unique_points = combined_points[unique_indices]

        # Mark new points (those not in total_points) as 1 in the trackable_filter
        for idx in unique_indices:
            if idx >= len(total_points):  # New points
                trackable_filter[idx] = 1

        # Slice trackable_filter to match the length of points
        trackable_filter = trackable_filter[len(total_points):]

        # Count new points
        new_points_len = np.sum(trackable_filter)

        return unique_points, trackable_filter, new_points_len


    def update_total_points_prev(self, points, prev_points, threshold=0.01):
        if prev_points.size == 0:
            # 이전 프레임에 점이 없으면 모든 점을 새로운 점으로 간주
            trackable_filter = np.ones(points.shape[0], dtype=int)
            return points, trackable_filter, points.shape[0]

        # Create KD-Tree for previous points
        tree = cKDTree(prev_points)

        # Find unique points in current frame
        unique_indices = []
        trackable_filter = np.zeros(points.shape[0], dtype=int)  # Initialize filter

        for i, point in enumerate(points):
            # Query points within the threshold
            neighbors = tree.query_ball_point(point, threshold)
            if len(neighbors) == 0:  # If no neighbors, it's a new point
                unique_indices.append(i)
                trackable_filter[i] = 1

        # Extract unique points
        unique_points = points[unique_indices]

        # Update prev_points by appending unique points
        updated_prev_points = np.vstack((prev_points, unique_points))

        # Count new points
        new_points_len = len(unique_indices)

        return updated_prev_points, trackable_filter, new_points_len
    
    def update_total_points_firstone(self, points, prev_points, threshold=0.01):
        if prev_points.size == 0:
            # 이전 프레임에 점이 없으면 모든 점을 새로운 점으로 간주
            # trackable_filter = np.zeros(points.shape[0], dtype=int)  # 모든 값을 0으로 초기화
            # trackable_filter[0] = 1  # 첫 번째 값만 1로 설정
            trackable_filter = np.ones(points.shape[0], dtype=int)
            return points, trackable_filter, points.shape[0]

        # Create KD-Tree for previous points
        tree = cKDTree(prev_points)

        # Find unique points in current frame
        unique_indices = []
        trackable_filter = np.zeros(points.shape[0], dtype=int)  # Initialize filter

        for i, point in enumerate(points):
            # Query points within the threshold
            neighbors = tree.query_ball_point(point, threshold)
            if len(neighbors) == 0:  # If no neighbors, it's a new point
                unique_indices.append(i)
                trackable_filter[i] = 1

        trackable_filter = np.zeros(points.shape[0], dtype=int)  # Initialize filter
        trackable_filter[0] = 1  # 첫 번째 값만 1로 설정

        # Extract unique points
        unique_points = points[unique_indices]

        # Update prev_points by appending unique points
        updated_prev_points = np.vstack((prev_points, unique_points))

        # Count new points
        new_points_len = len(unique_indices)



        return updated_prev_points, trackable_filter, new_points_len


    def update_total_points_parallel(self, points, total_points, threshold=0.01):
        if total_points.size == 0:
            # 모든 점이 새로운 점으로 간주되므로 trackable_filter는 모두 1
            trackable_filter = np.ones(points.shape[0], dtype=int)
            return points, trackable_filter, points.shape[0]

        # Combine new points with total_points
        combined_points = np.vstack((total_points, points))

        # Create KD-Tree for combined points
        tree = cKDTree(combined_points)

        # Define a function to process each point in parallel
        def find_unique(i):
            # Query points within the threshold
            neighbors = tree.query_ball_point(combined_points[i], threshold)
            return i, neighbors

        # Use ThreadPoolExecutor to process points in parallel
        unique_indices = []
        visited = set()
        trackable_filter = np.zeros(combined_points.shape[0], dtype=int)  # Initialize filter

        with ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(find_unique, i): i for i in range(combined_points.shape[0])}
            for future in future_to_index:
                i, neighbors = future.result()
                if i not in visited:
                    visited.update(neighbors)  # Mark neighbors as visited
                    unique_indices.append(i)  # Keep the first occurrence

        # Extract unique points
        unique_points = combined_points[unique_indices]

        # Mark new points (those not in total_points) as 1 in the trackable_filter
        for idx in unique_indices:
            if idx >= len(total_points):  # New points
                trackable_filter[idx] = 1

        # Slice trackable_filter to match the length of points
        trackable_filter = trackable_filter[len(total_points):]

        # Count new points
        new_points_len = np.sum(trackable_filter)

        return unique_points, trackable_filter, new_points_len
    

    def replace_points(self):
        points = torch.tensor([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=torch.float32)
        colors = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        rots = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=torch.float32)
        scales = torch.tensor([[0.5, 0.1, 0.1], [0.5, 0.1, 0.1], [0.5, 0.1, 0.1]], dtype=torch.float32)
        z_values = torch.tensor([1, 1, 1], dtype=torch.float32)
        trackable_filter = np.arange(0, points.shape[0], dtype=np.int64)

        return points, colors, rots, scales, z_values, trackable_filter
    

    def replace_points_zero(self):
        points = torch.tensor([], dtype=torch.float32)
        colors = torch.tensor([], dtype=torch.float32)
        rots = torch.tensor([], dtype=torch.float32)
        scales = torch.tensor([], dtype=torch.float32)
        z_values = torch.tensor([], dtype=torch.float32)
        trackable_filter = np.arange(0, 0, dtype=np.int64)

        return points, colors, rots, scales, z_values, trackable_filter


    def compute_gicp_covariances(self, pcd, k=10):
        """GICP 방식으로 covariance 행렬을 계산"""
        points = np.asarray(pcd.points)
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        covariances = []
        
        for i, p in enumerate(points):
            _, idxs, _ = kd_tree.search_knn_vector_3d(p, k)  # k개의 이웃 찾기
            neighbors = points[idxs, :]  # 이웃 점들
            
            # 평균 계산
            mean = np.mean(neighbors, axis=0)
            
            # 공분산 행렬 계산
            cov = np.cov((neighbors - mean).T, bias=True)  # (3xN) 형태로 변환 후 covariance 계산
            
            # 정규화된 covariance를 저장
            covariances.append(cov)
        
        return np.array(covariances)


    def gicp_data_association(self, source_points, target_points, T_ex):
        # 1) Open3D PointCloud로 변환
        source_pcd = o3d.geometry.PointCloud()
        target_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        
        # 3) GICP용 Covariance 행렬 계산 (전체 포인트 기준)
        source_covariances = self.compute_gicp_covariances(source_pcd, k=10)
        target_covariances = self.compute_gicp_covariances(target_pcd, k=10)
        
        # 4) 특정 거리(예: 50m) 이상인 점 제거를 위한 필터링
        source_distances = np.linalg.norm(source_points, axis=1)
        target_distances = np.linalg.norm(target_points, axis=1)

        valid_source_indices = source_distances < 10
        valid_target_indices = target_distances < 10

        # 원본 -> 필터링 후 인덱스로의 매핑을 위해,
        # 필터링 통과한 점들의 "원본 인덱스" 배열을 따로 구한다.
        source_indices_filtered = np.where(valid_source_indices)[0]  # 예: [0, 1, 4, 5, ...]
        target_indices_filtered = np.where(valid_target_indices)[0]

        # 실제 필터링된 좌표로 PointCloud 업데이트
        filtered_source_points = source_points[valid_source_indices]
        filtered_target_points = target_points[valid_target_indices]
        
        source_pcd.points = o3d.utility.Vector3dVector(filtered_source_points)
        target_pcd.points = o3d.utility.Vector3dVector(filtered_target_points)

        # 2) 노멀 추정
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # 5) 일반화된 ICP(GICP) 수행
        #    - init=T_ex 를 통해 초기 변환행렬 지정
        reg_p2p = o3d.pipelines.registration.registration_generalized_icp(
            source_pcd, 
            target_pcd, 
            max_correspondence_distance=30,
            init=T_ex,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-3, 
                relative_rmse=1e-3, 
                max_iteration=10
            )
        )
        
        # 6) correspondences와 최종 변환행렬 획득
        # reg_p2p.correspondence_set 은 (N,2) 형태의 NumPy 배열이며,
        # 각 행 = [필터링된 소스 점 인덱스, 필터링된 타깃 점 인덱스]
        correspondences_filtered = np.asarray(reg_p2p.correspondence_set)
        T_gicp = reg_p2p.transformation
        
        # 7) 필터링 후 인덱스를 다시 "원본 인덱스"로 역매핑
        #    correspondences_filtered[:, 0] 은 source_pcd(필터링된)에 대한 인덱스 -> source_indices_filtered를 사용
        #    correspondences_filtered[:, 1] 은 target_pcd(필터링된)에 대한 인덱스 -> target_indices_filtered를 사용
        original_source_corr_idx = source_indices_filtered[correspondences_filtered[:, 0]]
        original_target_corr_idx = target_indices_filtered[correspondences_filtered[:, 1]]
        
        # 가독성을 위해 (N,2) 형태로 다시 쌓는다.
        # 첫 번째 열: 원본 소스 인덱스, 두 번째 열: 원본 타깃 인덱스
        correspondences_in_original = np.column_stack((original_source_corr_idx, original_target_corr_idx))
        
        # 8) 필요한 정보를 리턴
        #    - 원본 인덱스 기준의 correspondences
        #    - 최종 추정된 GICP 변환행렬
        #    - (원본 사이즈 기준) GICP Covariances

        return correspondences_in_original, T_gicp, source_covariances, target_covariances

    

    def se3_exp_map(self, xi):
        omega = xi[:3]  # 회전 벡터 (so(3))
        v = xi[3:6]  # 병진 벡터 (R^3)
        
        # skew-symmetric matrix for omega
        Omega = np.array([[0, -omega[2], omega[1]],
                        [omega[2], 0, -omega[0]],
                        [-omega[1], omega[0], 0]])

        theta = np.linalg.norm(omega)
        
        if theta < 1e-8:  # 작은 회전 값에 대한 근사
            R = np.eye(3)
            V = np.eye(3)
        else:
            R = scipy.linalg.expm(Omega)  # so(3) → SO(3)
            V = np.eye(3) + (1 - np.cos(theta)) / (theta**2) * Omega + (theta - np.sin(theta)) / (theta**3) * np.dot(Omega, Omega)
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = V @ v  # 병진 변환

        return T


    def se3_log_map(self, T):
        R = T[:3, :3]
        t = T[:3, 3]
        
        # print("trR: ")
        # print (np.trace(R))

        trace_R = np.trace(R)
        theta = np.arccos(np.clip((trace_R - 1) / 2, -1.0, 1.0))
        
        if np.abs(theta) < 1e-8:
            omega = np.zeros(3)
            v = t
        else:
            lnR = (theta / (2 * np.sin(theta))) * (R - R.T)
            omega = np.array([lnR[2, 1], lnR[0, 2], lnR[1, 0]])
            
            A_inv = np.eye(3) - 0.5 * lnR + (1 - (theta * np.cos(theta) / (2 * np.sin(theta)))) / (theta**2) * lnR @ lnR
            v = A_inv @ t
        
        return np.hstack((omega, v))  # [ω_x, ω_y, ω_z, v_x, v_y, v_z]
    

    # def objective_function(self, xi, points, points_prev, corres_idx, cov_s, cov_t, pose_prop_curr, pose_opt_prev):       
    #     # 1. xi (Lie algebra) → SE(3) 변환 행렬 T 변환
    #     T = self.se3_exp_map(xi)

    #     # 2. 포인트 변환
    #     points_prev_homo = np.hstack((points_prev, np.ones((points_prev.shape[0], 1))))  # Nx4 호모지니어스 좌표
    #     transformed_points_prev = np.dot(T, points_prev_homo.T).T[:, :3]  # T 적용 후 변환된 포인트

    #     # 4. J_icp
    #     J_icp = 0
    #     for i in range(len(corres_idx)):  # 인덱스를 사용하여 루프 처리
    #         idx_src = corres_idx[i, 0]  # source point index
    #         idx_tgt = corres_idx[i, 1]  # target point index
    #         error = points[idx_src, :] - transformed_points_prev[idx_tgt, :]
    #         R = T[:3, :3]  # 회전 행렬
    #         covariance_term = cov_t[0] + R @ cov_s[0] @ R.T  # 공분산 행렬 계산
    #         J_icp += error.T @ np.linalg.inv(covariance_term) @ error  # 비용 함수 업데이트

    #     # 5. J_prop 
    #     pose_prev_inv = np.linalg.inv(pose_opt_prev)
    #     pose_curr = pose_prop_curr
    #     Log_T = self.se3_log_map(T @ pose_prev_inv @ pose_curr)

    #     J_prop = np.sum(Log_T[:3] @ Log_T[:3]) + np.sum(Log_T[3:6] @ Log_T[3:6])

    #     # 6. 최종 비용 함수
    #     cost = 0*J_icp + 1*J_prop

    #     return cost
    

    # def optimize_T(self, points, points_prev, corres_idx, cov_s, cov_t, pose_prop_curr, pose_opt_prev):

    #     # 초기 변환 행렬 T0 → Lie algebra 벡터 xi0로 변환
    #     # pose_prop_curr_inv = np.linalg.inv(pose_prop_curr)
    #     # xi0 = self.se3_log_map(pose_prop_curr_inv @ pose_opt_prev) + np.array([0, 0, 0, 1e-6, 1e-6, 1e-6])
    #     xi0 = np.zeros(6)  # 초기값 (0으로 시작)

    #     # 최적화 실행 (BFGS 사용)
    #     result = minimize(self.objective_function, xi0, args=(points, points_prev, corres_idx, cov_s, cov_t, pose_prop_curr, pose_opt_prev), method='BFGS')

    #     # 최적화된 SE(3) 변환 행렬 반환
    #     T_opt = self.se3_exp_map(result.x)

    #     # print("cost:", result.fun)

    #     return T_opt




    def compute_gicp_covariances(self, points, k=10):       
        N = len(points)
        covs = np.zeros((N, 3, 3), dtype=np.float64)
        
        kdtree = cKDTree(points)
        
        for i in range(N):
            # 자기 자신 포함 k+1개 이웃을 찾음
            dists, idxs = kdtree.query(points[i], k=k+1)
            # idxs[0]이 자기 자신이므로 제외
            neighbors = points[idxs[1:]]  # (k, 3)
            
            # 이웃들의 평균, 공분산 계산
            mu = neighbors.mean(axis=0)
            diff = neighbors - mu
            cov = diff.T @ diff / k  # (3,3)
            
            # 정칙성(singularity) 방지를 위해 작은 regularization
            cov += np.eye(3)*1e-6
            
            covs[i] = cov
        
        return covs


    def se3_exp(self, xi):
        """
        SE(3) Lie Algebra 원소 xi = [rx, ry, rz, tx, ty, tz] (길이 6 벡터)를
        4x4 변환행렬(회전+평행이동)로 변환한다.
        
        여기서는 소이각(small angle) 근사 또는 Rodrigues 공식을 간단히 사용.
        
        Parameters:
        -----------
        xi : (6,) shape numpy array
            [rx, ry, rz, tx, ty, tz]
        
        Returns:
        --------
        T : (4,4) numpy array
            3D 변환행렬
        """
        rx, ry, rz, tx, ty, tz = xi
        
        # 회전벡터의 크기(각도)
        theta = np.sqrt(rx*rx + ry*ry + rz*rz)
        
        # 기본값
        R = np.eye(3)
        
        if theta < 1e-12:
            # 매우 작은 회전 -> 근사 R ~ I
            pass
        else:
            # Rodrigues 공식
            kx, ky, kz = rx/theta, ry/theta, rz/theta
            K = np.array([[0,   -kz,  ky ],
                        [kz,   0,  -kx ],
                        [-ky,  kx,   0 ]])
            R = (
                np.eye(3) 
                + np.sin(theta)*K 
                + (1-np.cos(theta))*(K @ K)
            )
        
        T = np.eye(4)
        T[:3,:3] = R
        T[0,3] = tx
        T[1,3] = ty
        T[2,3] = tz
        
        return T


    def transform_points(self, T, points):
        N = len(points)
        hom = np.column_stack([points, np.ones(N)])  # (N,4)
        transformed = (T @ hom.T).T  # (N,4)
        return transformed[:, :3]


    def build_system_gicp(self, source_points, target_points, source_covs, target_covs, T):
        """
        GICP 최적화에 필요한 가우스-뉴턴 시스템(H, b)을 구성.
        
        - d_i = T * source[i] - target[i]
        - 가중치 W_i = (C_t + R * C_s * R^T)^(-1)
        - Gauss-Newton: sum_i [ J_i^T W_i J_i ], sum_i [ J_i^T W_i d_i ]
        
        Parameters:
        -----------
        source_points : (N,3) numpy array
        target_points : (N,3) numpy array
        source_covs   : (N,3,3) numpy array, 소스 각 점의 공분산
        target_covs   : (N,3,3) numpy array, 타깃 각 점의 공분산
        T : (4,4) numpy array, 현재 추정된 변환행렬
        
        Returns:
        --------
        H : (6,6) numpy array
        b : (6,)  numpy array
        """
        N = len(source_points)
        
        # 소스점 변환
        transformed_source = self.transform_points(T, source_points)  # (N,3)
        d = transformed_source - target_points  # 잔차 (N,3)
        
        R = T[:3,:3]
        
        # 미리 R * source_cov * R^T + target_cov 계산
        # 이들의 역행렬을 invM_i 라고 할 때, W_i = invM_i
        invM_list = []
        for i in range(N):
            M = target_covs[i] + R @ source_covs[i] @ R.T
            invM = np.linalg.inv(M)
            invM_list.append(invM)
        invM_list = np.stack(invM_list, axis=0)  # (N,3,3)
        
        # Gauss-Newton 시스템 구성
        H = np.zeros((6,6), dtype=np.float64)
        b = np.zeros((6,), dtype=np.float64)
        
        # R * p_s = transformed_source - 평행이동
        Rps = transformed_source - T[:3,3]  # (N,3)
        
        for i in range(N):
            invM = invM_list[i]    # (3,3)
            r_i  = d[i]            # (3,)
            px, py, pz = Rps[i]
            
            # 소각 근사에서,
            # Jr = d(d_i)/d(rx,ry,rz) = -[R * p_s]_x
            # Jt = d(d_i)/d(tx,ty,tz) = I
            Jr = np.array([[   0, -pz,  py],
                        [ pz,   0, -px],
                        [-py, px,   0]], dtype=np.float64)
            Jr = -Jr  # 부호
            
            Jt = np.eye(3)
            
            # J_i = [Jr, Jt]  => shape (3,6)
            J_i = np.hstack((Jr, Jt))  # (3,6)
            
            # A = J_i^T * invM  => (6,3)
            A = J_i.T @ invM
            # H_i = A @ J_i => (6,6)
            H_i = A @ J_i
            # b_i = A @ r_i => (6,)
            b_i = A @ r_i
            
            H += H_i
            b += b_i
        
        return H, b


    # def gicp_from_scratch_with_outlier_filter(
    #     self,
    #     source_points, 
    #     target_points,
    #     T_ex = np.eye(4),
    #     k=10,
    #     max_iter=20,
    #     epsilon=1e-6,
    #     dist_threshold=0.2
    # ):
        
    #     # 1) 공분산 계산
    #     source_covariances = self.compute_gicp_covariances(source_points, k=k)
    #     target_covariances = self.compute_gicp_covariances(target_points, k=k)
        
    #     # 2) 초기 변환
    #     T_gicp = T_ex.copy()
        
    #     # 3) KD-트리 (타깃) 생성
    #     kdtree = cKDTree(target_points)
        
    #     # (마지막) correspondences 저장 용도
    #     correspondences_in_original = None
        
    #     # 4) 루프
    #     for iter_idx in range(max_iter):
    #         # (A) 현재 T_gicp로 변환한 소스 -> 최근접 타깃 점 
    #         src_transformed = self.transform_points(T_gicp, source_points)
    #         dists, idxs = kdtree.query(src_transformed)  # dists: (N,), idxs: (N,)
            
    #         # (B) 거리 필터 -> outlier 제외
    #         valid_mask = dists < dist_threshold
    #         inlier_src_idx = np.where(valid_mask)[0]   # 소스 중에서 inlier
    #         inlier_tgt_idx = idxs[valid_mask]          # 타깃 인덱스
            
    #         # 소스 점, 타깃 점, 공분산도 inlier만 추출
    #         inlier_source_pts = source_points[inlier_src_idx]
    #         inlier_source_cov = source_covariances[inlier_src_idx]
    #         inlier_target_pts = target_points[inlier_tgt_idx]
    #         inlier_target_cov = target_covariances[inlier_tgt_idx]

    #         print ("corres: ")
    #         print (inlier_source_pts.shape[0])
    #         print (inlier_target_pts.shape[0])
            
    #         # 만약 유효점이 매우 적으면 break
    #         if len(inlier_source_pts) < 3:
    #             print(f"[GICP] Too few inliers (<3). Stop at iteration={iter_idx}")
    #             break
            
    #         # (C) Gauss-Newton 시스템
    #         H, b = self.build_system_gicp(
    #             inlier_source_pts,
    #             inlier_target_pts,
    #             inlier_source_cov,
    #             inlier_target_cov,
    #             T_gicp
    #         )
            
    #         try:
    #             delta_xi = -np.linalg.solve(H, b)
    #         except np.linalg.LinAlgError:
    #             print("[GICP] Singular matrix encountered, stopping.")
    #             break
            
    #         update_norm = np.linalg.norm(delta_xi)
    #         if update_norm < epsilon:
    #             print(f"[GICP] Converged at iteration={iter_idx}, update_norm={update_norm}, inliers={len(inlier_source_pts)}")
    #             correspondences_in_original = np.column_stack((inlier_src_idx, inlier_tgt_idx))
    #             break
            
    #         # (D) T 갱신
    #         dT = self.se3_exp(delta_xi)
    #         T_gicp = dT @ T_gicp
            
    #         # 매 반복마다 마지막에 inlier correspondence 기록
    #         correspondences_in_original = np.column_stack((inlier_src_idx, inlier_tgt_idx))
        
    #     # 최종
    #     return correspondences_in_original, T_gicp, source_covariances, target_covariances


    def objective_function(self, xi, points, points_prev, corres_idx, cov_s, cov_t, pose_prop_curr, pose_prop_prev):
        """
        (질문에서 주어진) 사용자 정의 목적함수:
          - ICP오차 (J_icp) + motion prior (J_prop)
          - 현재는 cost = 0*J_icp + 1*J_prop 구조
        """
        # 1) xi -> T
        T = self.se3_exp_map(xi)

        R = T[:3, :3]
        t = T[:3, 3]

        # 2) points_prev 변환
        points_prev_homo = np.hstack((points_prev, np.ones((points_prev.shape[0], 1))))  # (N,4)
        transformed_points_prev = (T @ points_prev_homo.T).T[:, :3]

        # 3) ICP cost (J_icp)
        J_icp = 0.0
        for i in range(len(corres_idx)):
            idx_src = corres_idx[i, 0]
            idx_tgt = corres_idx[i, 1]
            error = points[idx_src, :] - transformed_points_prev[idx_tgt, :]

            R = T[:3, :3]
            covariance_term = cov_t[idx_tgt] + R @ cov_s[idx_src] @ R.T  
            J_icp += error.T @ np.linalg.inv(covariance_term) @ error

        J_icp = 0.0
        for i in range(len(corres_idx)):
            idx_src = corres_idx[i, 0]
            idx_tgt = corres_idx[i, 1]
            error = points[idx_src, :] - transformed_points_prev[idx_tgt, :]
            covariance_term = cov_t[idx_tgt] + R @ cov_s[idx_src] @ R.T  
            J_icp += error.T @ error

        # 4) motion prior cost (J_prop)
        pose_prev_inv = np.linalg.inv(pose_prop_prev)
        Log_T = self.se3_log_map(T @ pose_prev_inv @ pose_prop_curr)

        # 임시: 회전/평행이동 norm^2
        J_prop = np.sum(Log_T[:3]**2) + np.sum(Log_T[3:]**2)

        # 5) 최종 cost
        cost = 1 * J_icp + 0 * J_prop

        return cost


    def optimize_T(self, points, points_prev, corres_idx, cov_s, cov_t, pose_prop_curr, pose_prop_prev):
        # 초기치 설정
        T_tw = np.linalg.inv(pose_prop_curr)
        xi0 = self.se3_log_map(T_tw @ pose_prop_prev) + np.array([0, 0, 0, 1e-6, 1e-6, 1e-6])

        # BFGS
        result = minimize(
            self.objective_function, 
            xi0, 
            args=(points, points_prev, corres_idx, cov_s, cov_t, pose_prop_curr, pose_prop_prev), 
            method='BFGS'
        )
        
        T_opt = self.se3_exp_map(result.x)


        print("cost:", result.fun)

        T = self.se3_exp_map(xi0)
        points_prev_homo = np.hstack((points_prev, np.ones((points_prev.shape[0], 1))))  # (N,4)
        transformed_points_prev = (T @ points_prev_homo.T).T[:, :3]
        J_icp = 0.0
        for i in range(len(corres_idx)):
            idx_src = corres_idx[i, 0]
            idx_tgt = corres_idx[i, 1]
            error = points[idx_src, :] - transformed_points_prev[idx_tgt, :]
            J_icp += error.T @ error
        print("cost_start:", J_icp)

        return T_opt


    def gicp_bfgs_with_outlier_filter(
        self,
        source_points, 
        target_points,
        T_ex,
        k,
        corres_num_threshold,
        dist_threshold,
        pose_prop_curr,
        pose_prop_prev
    ):
        # 1) 공분산
        source_covariances = self.compute_gicp_covariances(source_points, k=k)
        target_covariances = self.compute_gicp_covariances(target_points, k=k)

        # 2) KD-트리
        kdtree = cKDTree(target_points)

        # 3) 초기 변환 T_ex로 소스 변환
        src_transformed = self.transform_points(T_ex, source_points)

        # 4) 최근접점 + Outlier 제거
        dists, idxs = kdtree.query(src_transformed)  # (N,), (N,)
        valid_mask = dists < dist_threshold
        inlier_src_idx = np.where(valid_mask)[0]
        inlier_tgt_idx = idxs[valid_mask]

        print("lenlen: ", len(inlier_src_idx))

        # # -> (X,)개
        # if len(inlier_src_idx) < corres_num_threshold:
        #     print("[GICP BFGS] Too few inliers -> return T_ex")
        #     return T_ex

        # correspondences_in_original: (X,2)
        correspondences_in_original = np.column_stack((inlier_src_idx, inlier_tgt_idx))
        print(inlier_tgt_idx.shape[0])

        # 5) BFGS 최적화
        #  - objective_function은 (source_points, target_points, ...)을 args로 받음
        #  - 여기서 "points" = source_points, "points_prev" = target_points로 지칭
        #  - cov_s, cov_t도 "점별"이라면 여기서 index 맞춰야 하지만, 질문 예시는 [0], [0]만 쓰는 구조
        #    (실제로는 점별로 써야 함)
        T_opt = self.optimize_T(
            points = source_points,
            points_prev = target_points,
            corres_idx = correspondences_in_original,
            cov_s = source_covariances,
            cov_t = target_covariances,
            pose_prop_curr = pose_prop_curr,
            pose_prop_prev  = pose_prop_prev
        )

        return T_opt


        #LYS
    def visualize_pointcloud_with_colors(self, points, colors):
        """
        points: (N, 3) shape NumPy array (float)
        colors: (N, 3) shape NumPy array (float), each value in [0, 1]
        """
        # Open3D PointCloud 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)   # Nx3 float
        pcd.colors = o3d.utility.Vector3dVector(colors)   # Nx3 float, [0,1]

        # 시각화
        o3d.visualization.draw_geometries([pcd])
        
    def visualize_depth_original(self, depth_img, depth_scale=800.0):
        """
        depth_img : (H, W) 형태의 원본 depth 이미지 (정수/float)
        depth_scale : depth를 실제 단위(m 등)로 만들기 위한 스케일
        """
        # float 변환 및 단위 스케일 적용
        depth_m = depth_img.astype(np.float32) / depth_scale
        
        # 시각화 (matplotlib)
        plt.imshow(depth_m, cmap='gray')
        plt.colorbar(label="Depth (m)")
        plt.title("Original Depth Image")
        plt.show()
