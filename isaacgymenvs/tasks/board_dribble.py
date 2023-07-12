import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask

class Board_dribble(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.max_episode_length = 1000

        # taskごとに設定する箇所
        self.reset_dist = self.cfg["env"]["resetDist"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]
       
        # for object
        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["ball"]
        self.asset_files_dict = {
            "ball": "urdf/objects/ball.urdf",
        }
        if "asset" in self.cfg["env"]:
            self.asset_files_dict["ball"] = self.cfg["env"]["asset"].get("assetFileNameBall", self.asset_files_dict["ball"])
        
        # arm_dof_pos(3) + arm_dof_vel(3) + ball_pos(3) + ball_vel(3) + action(3)
        self.cfg["env"]["numObservations"] = 15
        self.cfg["env"]["numActions"] = 3

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # get gym GPU state tensors
        # for robot
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        
        # for object
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) 

        # 剛体として計算するものに使ってるが、イマイチ何に必要かわからない
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        # -1の所は他の引数から決まる = dofs of robot
        # num_envs * dofs of robot * 2(pos,vel) = dof_state_tensorの要素数
        self.arm_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_dofs]
        self.arm_dof_pos = self.dof_state.view(self.num_envs, self.num_arm_dofs, 2)[..., 0]
        self.arm_dof_vel = self.dof_state.view(self.num_envs, self.num_arm_dofs, 2)[..., 1]

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
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
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
        arm_asset = self.gym.load_asset(self.sim, asset_root, arm_asset_file, asset_options)
        asset_options.vhacd_enabled = True
        asset_options.use_mesh_materials = True
        
        self.num_arm_bodies = self.gym.get_asset_rigid_body_count(arm_asset)
        self.num_arm_shapes = self.gym.get_asset_rigid_shape_count(arm_asset)
        self.num_arm_dofs = self.gym.get_asset_dof_count(arm_asset) # 3
        
        arm_parts_names = [self.gym.get_asset_rigid_body_name(arm_asset, i) for i in range(self.num_arm_bodies)]
        sensor_attach_names = [s for s in arm_parts_names if "hand" in s]
        self.sensor_attach_index = torch.zeros(len(sensor_attach_names), dtype=torch.long, device=self.device)

        # create force sensors attached to the top of surface
        sensor_attach_indices = [self.gym.find_asset_rigid_body_index(arm_asset, name) for name in sensor_attach_names]
        sensor_pose = gymapi.Transform()
        for body_idx in sensor_attach_indices:
            self.gym.create_asset_force_sensor(arm_asset, body_idx, sensor_pose)

        # get board dof properties, loaded by Isaac Gym from URDF
        arm_dof_props = self.gym.get_asset_dof_properties(arm_asset)

        arm_start_pose = gymapi.Transform()
        arm_start_pose.p.z = 1.0
        arm_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.5 * np.pi) # y軸方向に90°
        
        # load object
        object_asset_options = gymapi.AssetOptions()
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        object_asset_options.use_mesh_materials = True
        object_asset_options.vhacd_enabled = True
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z = 0.8, 0, 1.5

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
            arm_handle = self.gym.create_actor(env_ptr, arm_asset, arm_start_pose, "arm", i, 1, 0)
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
                 0, 0, 0, 0, 0, 0])
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
        
        self.arm_init_states = to_torch(self.arm_init_states, device=self.device).view(self.num_envs, 13)
        self.object_init_states = to_torch(self.object_init_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)

        self.arm_indices = to_torch(self.arm_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        # retrieve environment observations from buffer
        arm_pos = self.obs_buf[:, 0:3]
        arm_vel = self.obs_buf[:, 3:6]
        ball_pos = self.obs_buf[:, 6:9]
        ball_vel = self.obs_buf[:, 9:12]
        # success_count = self.obs_buf[:, 9]
        # print("success_count --", success_count)
        
        # force_value = self.obs_buf[:, 5:11]
      
        self.rew_buf[:], self.reset_buf[:] = compute_board_dribble_reward(
            arm_pos,
            arm_vel,
            ball_pos,
            ball_vel,
            # force_value,
            actions,
            # success_count,
            # sim_time,
            # pre_collision_time,
            self.reset_dist,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_vel = self.root_state_tensor[self.object_indices, 7:10]
        
        self.obs_buf[:, 0:3] = self.arm_dof_pos[:, 0:3].squeeze()
        self.obs_buf[:, 3:6] = self.arm_dof_vel[:, 0:3].squeeze()
        self.obs_buf[:, 6:9] = self.object_pos[:, 0:3].squeeze()
        self.obs_buf[:, 9:12] = self.object_vel[:, 0:3].squeeze()
        self.obs_buf[:, 12:15] = self.actions
       
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        sensors_per_env = 1
        force_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        
        return self.obs_buf

    def reset_idx(self, env_ids):
        # reset arm pos/vel
        arm_positions = 0.1 * (torch.rand((len(env_ids), self.num_arm_dofs), device=self.device) - 0.5)
        arm_velocities = 0.1 * (torch.rand((len(env_ids), self.num_arm_dofs), device=self.device))

        self.arm_dof_pos[env_ids, :] = arm_positions[:]
        self.arm_dof_vel[env_ids, :] = arm_velocities[:]
       
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
        forces = self.actions * self.max_push_effort
        force_tensor = gymtorch.unwrap_tensor(forces)
        # print("force_tensor", force_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        

        # print("hit time?", self.gym.get_sim_time(self.sim))
        # print("hit time?", self.gym.get_sim_time(self.sim))
        # print("dof_state_tensor----", self.dof_state)
        # print("board_dof_state----", self.board_dof_state)
        # print("num_dofs-----------" , self.num_dofs)
        # print("board_dof_pos----", self.board_dof_pos)
        # print("board_dof_vel----", self.board_dof_vel)
        # actions_tensor = torch.zeros(self.num_envs * self.num_board_dofs, device=self.device, dtype=torch.float)
        # actions_tensor[::self.num_board_dofs] = actions.to(self.device).squeeze() * self.max_push_effort
        # print("actions_tensor", actions_tensor)
        # forces = gymtorch.unwrap_tensor(actions_tensor)

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
def compute_board_dribble_reward(
        arm_pos,
        arm_vel,
        ball_pos,
        ball_vel,
        # force_value,
        actions,
        # sim_time,
        # pre_collision_time,
        reset_dist,
        reset_buf,
        progress_buf,
        max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    ball_x = ball_pos[:, 0]
    ball_y = ball_pos[:, 1]
    ball_z = ball_pos[:, 2]
    
    # # ball height reward
    # height_reward = 0.8 * ball_z
    # # print("height_reward-----", height_reward)

    # target_height = 4.0
    target_height_reward = 16.0 - abs(4.0 - ball_z)**2
    
    # x vel penalty
    x_vel_minus_reward = - 0.1 * abs(ball_vel[:, 0])**2
    # print("x_vel_minus_reward-----", x_vel_minus_reward)
    
    # energy penalty for movement
    action_cost = 0.3 * torch.sum(actions ** 2, dim=-1)
    # print("action_cost----", action_cost)

    # total reward
    # total_reward = height_reward - action_cost + x_vel_minus_reward
    total_reward = target_height_reward - action_cost + x_vel_minus_reward
    # print("total_reward----", total_reward)
    
    # reward = torch.where(torch.abs(err_dist) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    # torch.whereは１つ１つ比較する
    reset = reset_buf
    reset = torch.where(abs(ball_x - 0.8) > 1.7, torch.ones_like(reset_buf), reset_buf) # x
    reset = torch.where(ball_z < 0.15, torch.ones_like(reset_buf), reset_buf) # z
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return total_reward, reset
