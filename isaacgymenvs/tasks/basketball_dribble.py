import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask

class BasketballDribble(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self.cfg = cfg
        self.max_episode_length = 1000

        # taskごとに設定する箇所
        self.max_push_effort = self.cfg["env"]["maxEffort"] # 200[Nm]

        # plane param
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
       
        # for object
        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["basketball"]
        self.asset_files_dict = {
            "basketball": "urdf/objects/basketball.urdf",
        }
        if "asset" in self.cfg["env"]:
            self.asset_files_dict["basketball"] = self.cfg["env"]["asset"].get("assetFileNameBall", self.asset_files_dict["basketball"])
        
        #   arm_dof_pos(3) + arm_dof_vel(3)
        # + ball_pos(3) + ball_rot(4) + ball_vel(3)
        # + force_sensor(6)
        # + action(3)
        # = 25
        self.cfg["env"]["numObservations"] = 25
        self.cfg["env"]["numActions"] = 3

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # for reset timing
        self.zero_vel_time = torch.zeros(self.num_envs).to(self.device)
        self.zero_vel_start_time = torch.zeros(self.num_envs).to(self.device)
        
        # get gym GPU state tensors
        # アドレスの形で取得 -> wrap_tensorで計算できる形に変換する必要あり
        # for robot
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # for object
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        # 剛体として計算するものに使ってるが、イマイチ何に必要かわからない
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # refresh
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # [dofs, 2]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13) # 13

        # for robot
        # -1の所は他の引数から決まる = dofs of robot
        # num_envs * dofs of robot * 2(pos,vel) = dof_state_tensorの要素数
        self.arm_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_dofs] # num_envs * [dofs, 2]

        self.arm_dof_pos = self.dof_state.view(self.num_envs, self.num_arm_dofs, 2)[..., 0] # num_envs * dofs
        self.arm_dof_vel = self.dof_state.view(self.num_envs, self.num_arm_dofs, 2)[..., 1] # num_envs * dofs

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]        
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)
        
    def create_sim(self):
        self.up_axis = self.cfg["sim"]["up_axis"] # z
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # 環境のスペースの設定
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, 4 * spacing)
        
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        arm_asset_file = "urdf/arm.urdf"
        object_asset_file = self.asset_files_dict[self.object_type]
        
        if "asset" in self.cfg["env"]:
            arm_asset_file = self.cfg["env"]["asset"].get("assetFileName", arm_asset_file)
            
        # load arm_asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        # asset_options.vhacd_enabled = True
        # asset_options.use_mesh_materials = True
        arm_asset = self.gym.load_asset(self.sim, asset_root, arm_asset_file, asset_options)
        
        self.num_arm_bodies = self.gym.get_asset_rigid_body_count(arm_asset)
        self.num_arm_shapes = self.gym.get_asset_rigid_shape_count(arm_asset)
        self.num_arm_dofs = self.gym.get_asset_dof_count(arm_asset) # 3
        
        # get board dof properties, loaded by Isaac Gym from URDF
        arm_dof_props = self.gym.get_asset_dof_properties(arm_asset)

        # define dof_limits
        self.arm_dof_lower_limits = []
        self.arm_dof_upper_limits = []
        for i in range(self.num_arm_dofs):
            self.arm_dof_lower_limits.append(arm_dof_props['lower'][i])
            self.arm_dof_upper_limits.append(arm_dof_props['upper'][i])

        self.arm_dof_lower_limits = to_torch(self.arm_dof_lower_limits, device=self.device)
        self.arm_dof_upper_limits = to_torch(self.arm_dof_upper_limits, device=self.device)

        # sensor取り付け場所の設定
        arm_parts_names = [self.gym.get_asset_rigid_body_name(arm_asset, i) for i in range(self.num_arm_bodies)]
        sensor_attach_names = [s for s in arm_parts_names if 'hand' in s] # 'hand'
        self.sensor_attach_index = torch.zeros(len(sensor_attach_names), dtype=torch.long, device=self.device) # sensor_attach_namesの数だけ0が並んだテンソルを生成

        # create force sensors attached to the top of surface
        sensor_attach_indices = [self.gym.find_asset_rigid_body_index(arm_asset, name) for name in sensor_attach_names] # assetファイルの中で何番目のrigid_bodyに取り付けるか -> indices = 3
        sensor_pose = gymapi.Transform()
        # handの原点をjoin部分に設定しているので、z方向に0.15ずらす必要あり
        # todo: ここを手動で変えずにファイルから読み出せるようにする
        sensor_pose.p.z = 0.15
        # 同じ物体に取り付けられたセンサにおいてforceは同じ値を示すが、torqueはセンサーのローカル座標系基準で測定される
        for body_idx in sensor_attach_indices:
            self.gym.create_asset_force_sensor(arm_asset, body_idx, sensor_pose)
            
        # default start pose
        arm_start_pose = gymapi.Transform()
        arm_start_pose.p.z = 1.6
        arm_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.5 * np.pi) # y軸方向に90[degree]
        
        # load object
        object_asset_options = gymapi.AssetOptions()
        # object_asset_options.use_mesh_materials = True
        # object_asset_options.vhacd_enabled = True
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        # default start pose
        # pos
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x, object_start_pose.p.y = 0.7, 0
        # object_start_pose.p.z = arm_start_pose.p.z - 0.25 # - (radius + α)
        object_start_pose.p.z = 1.2
        # vel
        object_start_vel = gymapi.Velocity()
        object_start_vel.linear = gymapi.Vec3()
        object_start_vel.linear.x, object_start_vel.linear.y = 0.0, 0.0
        object_start_vel.linear.z = 0.0
        
        # print("object_start_pose", object_start_pose)

        self.arm_handles = []
        self.envs = []

        self.arm_init_states = []
        self.object_init_states = []
        
        self.arm_indices = []
        self.object_indices = []

        arm_rb_count = self.gym.get_asset_rigid_body_count(arm_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(arm_rb_count, arm_rb_count + object_rb_count))
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # arm
            arm_handle = self.gym.create_actor(env_ptr, arm_asset, arm_start_pose, "arm", i, -1, 0)
            # -1を使うとassetファイルのcollision設定を使ってくれる？
            self.arm_init_states.append(
                [arm_start_pose.p.x, arm_start_pose.p.y, arm_start_pose.p.z,
                 arm_start_pose.r.x, arm_start_pose.r.y, arm_start_pose.r.z, arm_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, arm_handle, arm_dof_props)
            arm_idx = self.gym.get_actor_index(env_ptr, arm_handle, gymapi.DOMAIN_SIM)
            self.arm_indices.append(arm_idx)

            # ball
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "ball", i, 0, 0)
            self.object_init_states.append(
                [object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                 object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                 object_start_vel.linear.x, object_start_vel.linear.y, object_start_vel.linear.z,
                 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            
            #dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT # arm
            #dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE # ball
            arm_dof_props['stiffness'][:] = 0.0
            arm_dof_props['damping'][:] = 0.0

            self.envs.append(env_ptr)
            self.arm_handles.append(arm_handle)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)

        # to_torch
        self.arm_init_states = to_torch(self.arm_init_states, device=self.device).view(self.num_envs, 13)
        self.object_init_states = to_torch(self.object_init_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)

        self.arm_indices = to_torch(self.arm_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        sim_time = self.gym.get_sim_time(self.sim)
        
        self.rew_buf[:], self.reset_buf[:], self.zero_vel_time, self.zero_vel_start_time = compute_dribble_reward(
            self.obs_buf,
            sim_time,
            self.zero_vel_time,
            self.zero_vel_start_time,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length
        )

        # print("zero_vel_time", self.zero_vel_time)
        
    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # sensor
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        sensors_per_env = 1
        force_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        
        # object
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_vel = self.root_state_tensor[self.object_indices, 7:10]

        # set
        self.obs_buf[:, 0:3] = self.arm_dof_pos[:, 0:3].squeeze()
        self.obs_buf[:, 3:6] = self.arm_dof_vel[:, 0:3].squeeze()
        self.obs_buf[:, 6:9] = self.object_pos[:, 0:3].squeeze()
        self.obs_buf[:, 9:13] = self.object_rot[:, 0:4].squeeze()
        self.obs_buf[:, 13:16] = self.object_vel[:, 0:3].squeeze()
        self.obs_buf[:, 16:22] = force_sensor_tensor
        self.obs_buf[:, 22:25] = self.actions

        return self.obs_buf

    def reset_idx(self, env_ids):
        # reset arm pos/vel
        joint_tensor = torch.tensor([1.0, -1.0, -0.4])
        arm_dof_pos = joint_tensor.repeat(len(env_ids), 1).to(self.device)
        arm_dof_vel = torch.zeros((len(env_ids), self.num_arm_dofs), device=self.device)

        self.arm_dof_pos[env_ids, :] = arm_dof_pos[:]
        self.arm_dof_vel[env_ids, :] = arm_dof_vel[:]
    
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32 * 2), len(env_ids_int32))
        # reset object state
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_states[env_ids].clone()
       
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(env_ids_int32 * 2 + 1), len(env_ids_int32))
        # env_ids
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        forces = self.actions * self.max_push_effort # force = torque
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        
    def post_physics_step(self):
        self.progress_buf += 1
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            
        self.compute_observations()
        self.compute_reward(self.actions)

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_dribble_reward(
        obs_buf,
        sim_time,
        pre_zero_vel_time, # 1step前のself.zero_vel_time
        zero_vel_start_time,
        reset_buf,
        progress_buf,
        max_episode_length
):
    # type: (Tensor, float, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    # set value
    arm_dof_pos = obs_buf[:, 0:3]
    arm_dof_vel = obs_buf[:, 3:6]
    ball_pos = obs_buf[:, 6:9]
    ball_rot = obs_buf[:, 9:13]
    ball_vel = obs_buf[:, 13:16]
    force_sensor = obs_buf[:, 16:22]
    actions = obs_buf[:, 22:25]

    ball_x = ball_pos[:, 0]
    ball_y = ball_pos[:, 1]
    ball_z = ball_pos[:, 2]

    # reset timing
    # 速度は完全に0.0にはならないことに注意
    # update zero_vel_start_time
    zero_vel_start_time = torch.where(zero_vel_start_time == 0.0, sim_time, zero_vel_start_time)
    zero_vel_start_time = torch.where(ball_vel[:, 2] < 1.0e-06, zero_vel_start_time, 0.0) # time or 0
    # print("zero_vel_start_time", zero_vel_start_time[1])
    
    # calc zero_vel_time
    zero_vel_time = sim_time - zero_vel_start_time
    zero_vel_time = torch.where(zero_vel_time == sim_time, 0.0, zero_vel_time) # time or 0
    # print("zero_vel_time", zero_vel_time[1])

    # calc reward
    # target_height = 1.5
    target_height_reward =  2.25 - abs(2.25 - ball_z)**2
    # 最初はz方向の速度を増やすことをメインのrewardにする
    if sim_time < 1.0:
        target_height_reward = torch.zeros_like(target_height_reward)
    # print("target_height_reward---", target_height_reward)

    # ballがhandより上にあることに対するpenalty
    # hand部分の高さが必要
    # ball_hand_dist = ball_z -

    # z_vel reward
    z_vel_reward = 2.0 * abs(ball_vel[:, 2])**2
    
    # x_vel penalty
    x_vel_minus_reward = - abs(ball_vel[:, 0])**2
    # print("x_vel_minus_reward---", x_vel_minus_reward)
    
    # energy penalty for movement
    action_cost = 0.1 * torch.sum(actions ** 2, dim=-1)
    # print("action_cost---", action_cost)
   
    # total reward
    # total_reward = target_height_reward - action_cost + x_vel_minus_reward + z_vel_reward
    total_reward = target_height_reward + x_vel_minus_reward + z_vel_reward
    # print("total_reward---", total_reward)
    
    # torch.whereは１つ１つ比較する
    reset = torch.where(abs(ball_x - 0.75) > 1.5, torch.ones_like(reset_buf), reset_buf) # x
    reset = torch.where(torch.abs(zero_vel_time) > 0.75, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    # reset zero_vel_time, zero_vel_start_time
    zero_vel_start_time = torch.where(torch.abs(zero_vel_time) > 1.0, torch.zeros_like(zero_vel_start_time), zero_vel_start_time)
    zero_vel_time = torch.where(torch.abs(zero_vel_time) > 1.0, torch.zeros_like(zero_vel_time), zero_vel_time)
    # print("zero_vel_start_time", zero_vel_start_time[1])
    # print("zero_vel_time", zero_vel_time[1])

    return total_reward, reset, zero_vel_time, zero_vel_start_time
