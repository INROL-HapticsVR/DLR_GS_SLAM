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

class Tracker(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = slam.keyframe_th
        self.knn_max_distance = slam.knn_max_distance
        self.overlapped_th = slam.overlapped_th
        self.overlapped_th2 = slam.overlapped_th2
        self.downsample_rate = slam.downsample_rate
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
        self.gaussian_init_scale = slam.gaussian_init_scale
        self.topic_num = slam.topic_num
        self.max_fps = slam.max_fps
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
        
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.gaussian_init_scale)

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
        
        self.num_images = 2000
        self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
        self.reg.set_max_knn_distance(self.knn_max_distance)
        if_mapping_keyframe = False

        self.total_start_time = time.time()
        pbar = tqdm(total=2000)

        ii = 0
        # points_total = np.empty((0, 3))
        points_prev = np.empty((0, 3))
        # for ii in range(self.num_images):
        while ii < 2000:
            self.iter_shared[0] = ii

            rgb_index, depth_index, pose_index, current_image, depth_image, R_wc, T_wc, T_depth = self.get_sharedmemory()
            
            # #For Debug drift
            # depth_image = 1000*np.ones((720, 1280), dtype=np.float32)
            # R_wc = np.array([
            #     [np.cos(2 * np.pi / 8 * ii), -np.sin(2 * np.pi / 8 * ii), 0],
            #     [np.sin(2 * np.pi / 8 * ii),  np.cos(2 * np.pi / 8 * ii), 0],
            #     [0, 0, 1]
            # ])
            # T_wc = np.array([0, 0, 0])
            # T_depth = np.array([0, 0, 0])
            # #For Debug drift
            
            # print("ddddd111")
            # print(rgb_index)
            # print(depth_index)
            # print(pose_index)
                
            # Make pointcloud
            points, colors, z_values, trackable_filter = self.downsample_and_make_pointcloud2(depth_image, current_image)
            # GICP
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
                        rr.Transform3D(translation=T_wc,
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

                # points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                points = np.matmul(R_wc, points.transpose()).transpose() + T_wc

                # Set initial pointcloud to target points
                self.reg.set_input_target(points)
                
                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                
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
                    rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.004*self.gaussian_init_scale))
            else:
                self.reg.set_input_source(points)
                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                self.reg.set_source_filter(num_trackable_points, input_filter)
                
                initial_pose = self.poses[-1]

                current_pose = self.reg.align(initial_pose)
                self.poses.append(current_pose)

                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose)
                # points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                points = np.matmul(R_wc, points.transpose()).transpose() + T_wc

                # For Debug 포인트 정합

                # points_total, trackable_filter, points_new_len = self.update_total_points_parallel(points, points_total, 0.01)
                points_prev, trackable_filter, points_new_len = self.update_total_points_prev(points, points_prev, 0.01)
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
                        rr.Transform3D(translation = T_wc,
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

                if  (self.iteration_images >= self.num_images-1 \
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

                    #For Debug 1
                    
                    # Add new gaussians
                    # Erase overlapped points from current pointcloud before adding to map gaussian #
                    # Using filter
                    # not_overlapped_indices_of_trackable_points = self.eliminate_overlapped2(distances, self.overlapped_th2) # 5e-5 self.overlapped_th
                    # trackable_filter = trackable_filter[not_overlapped_indices_of_trackable_points]          

                    points_new, colors_new, rots_new, scales_new, z_values_new \
                        = self.filter_new_values(points, colors, rots, scales, z_values, trackable_filter)
                    trackable_filter = np.arange(0, points_new_len, dtype=np.int64)
                    self.shared_new_gaussians.input_values(torch.tensor(points_new), torch.tensor(colors_new), 
                                                       torch.tensor(rots_new), torch.tensor(scales_new), 
                                                       torch.tensor(z_values_new), torch.tensor(trackable_filter))
                    
                    if self.rerun_viewer:
                        # rr.set_time_sequence("step", self.iteration_images)
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                        rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points_new, colors=colors_new, radii=0.004*self.gaussian_init_scale))
                    
                    #For Debug 1

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

                    # For Debug 3
                    points_new, colors_new, rots_new, scales_new, z_values_new \
                        = self.filter_new_values(points, colors, rots, scales, z_values, trackable_filter)
                    trackable_filter = np.arange(0, points_new_len, dtype=np.int64)
                    self.shared_new_gaussians.input_values(torch.tensor(points_new), torch.tensor(colors_new), 
                                                       torch.tensor(rots_new), torch.tensor(scales_new), 
                                                       torch.tensor(z_values_new), torch.tensor(trackable_filter))
                    
                    # self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                    #                                    torch.tensor(rots), torch.tensor(scales), 
                    #                                    torch.tensor(z_values), torch.tensor(trackable_filter))
                    # For Debug 3
                    
                    # Add new keyframe
                    depth_image = depth_image.astype(np.float32)/self.depth_scale
                    # self.shared_cam.setup_cam(R, T, current_image, depth_image)
                    self.shared_cam.setup_cam(R_wc, T_depth, current_image, depth_image)
                    self.shared_cam.cam_idx[0] = self.iteration_images
                    
                    self.is_mapping_keyframe_shared[0] = 1
            pbar.update(1)
            
            while 1/((time.time() - self.total_start_time)/(self.iteration_images+1)) > self.max_fps:    #30. float(self.test)
                time.sleep(1e-15)
                
            self.iteration_images += 1
            ii = ii+1
        
        # Tracking end
        pbar.close()
        self.final_pose[:,:,:] = torch.tensor(self.poses).float()
        self.end_of_dataset[0] = 1
        
        print(f"System FPS: {1/((time.time()-self.total_start_time)/self.num_images):.2f}")
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
                    T_wc = R_wu @ tmp[0:3]
                    T_depth = -np.matmul(R_wc.transpose(), T_wc)

            # Check if all indices are synchronized
            # if rgb_index > 275:
            #     while 1: time.sleep(1e-1) 

            if rgb_index == depth_index == pose_index and rgb_locker == depth_locker == pose_locker == 1:
                break

        # Return global index and synchronized data
        return rgb_index, depth_index, pose_index, rgb_image, depth_image, R_wc, T_wc, T_depth


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
