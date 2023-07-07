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
        self.max_episode_length = 500

        # taskごとに設定する箇所
        self.reset_dist = self.cfg["env"]["resetDist"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]
        
        self.success_count = 0
       
        # for object
        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["ball"]
        self.asset_files_dict = {
            "ball": "urdf/objects/ball.urdf",
        }
        if "asset" in self.cfg["env"]:
            self.asset_files_dict["ball"] = self.cfg["env"]["asset"].get("assetFileNameBall", self.asset_files_dict["ball"])
        
        # board_pos(1) + board_vel(1) + ball_pos(3) + force(6) + time(1) + action(1)
        # ballの状態は取得できる？
        self.cfg["env"]["numObservations"] = 13
        self.cfg["env"]["numActions"] = 1

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # self.hit_time_handle = self.gym.create_performance_timers(self.sim)
        
        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        
        self.board_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_board_dofs]
        self.board_dof_pos = self.dof_state.view(self.num_envs, self.num_board_dofs, 2)[..., 0]
        self.board_dof_vel = self.dof_state.view(self.num_envs, self.num_board_dofs, 2)[..., 1]
        
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

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
        lower = gymapi.Vec3(- 2 * spacing, -spacing, 0.0)
        upper = gymapi.Vec3(2 * spacing, spacing, 4 * spacing)
        
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        board_asset_file = "urdf/board.urdf"
        object_asset_file = self.asset_files_dict[self.object_type]
        
        if "asset" in self.cfg["env"]:
            board_asset_file = self.cfg["env"]["asset"].get("assetFileName", board_asset_file)
            
        # load board_asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        board_asset = self.gym.load_asset(self.sim, asset_root, board_asset_file, asset_options)
        # asset_options.use_mesh_materials = True
        asset_options.vhacd_enabled = True
        
        self.num_board_bodies = self.gym.get_asset_rigid_body_count(board_asset)
        self.num_board_shapes = self.gym.get_asset_rigid_shape_count(board_asset)
        self.num_board_dofs = self.gym.get_asset_dof_count(board_asset)
        board_parts_names = [self.gym.get_asset_rigid_body_name(board_asset, i) for i in range(self.num_board_bodies)]

        sensor_attach_names = [s for s in board_parts_names if "board" in s]
        self.sensor_attach_index = torch.zeros(len(sensor_attach_names), dtype=torch.long, device=self.device)

        # create force sensors attached to the top of surface
        sensor_attach_indices = [self.gym.find_asset_rigid_body_index(board_asset, name) for name in sensor_attach_names]
        sensor_pose = gymapi.Transform()
        for body_idx in sensor_attach_indices:
            self.gym.create_asset_force_sensor(board_asset, body_idx, sensor_pose)

        # get board dof properties, loaded by Isaac Gym from URDF
        board_dof_props = self.gym.get_asset_dof_properties(board_asset)

        pose = gymapi.Transform()
        pose.p.z = 0.3
        

        # load object
        object_asset_options = gymapi.AssetOptions()
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        # object_asset_options.use_mesh_materials = True
        object_asset_options.vhacd_enabled = True
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z = -0.1, 0, 1.0

        # self.num_ball_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        # self.num_ball_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        # self.num_ball_dofs = self.gym.get_asset_dof_count(object_asset)
        # self.num_ball_bodies = self.gym.get_asset_rigid_body_count(object_asset)
    
        self.board_handles = []
        self.envs = []
        
        self.board_indices = []
        self.object_indices = []

        board_rb_count = self.gym.get_asset_rigid_body_count(board_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(board_rb_count, board_rb_count + object_rb_count))
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # board
            board_handle = self.gym.create_actor(env_ptr, board_asset, pose, "board", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, board_handle, board_dof_props)
            board_idx = self.gym.get_actor_index(env_ptr, board_handle, gymapi.DOMAIN_SIM)
            self.board_indices.append(board_idx)

            # ball
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "ball", i, 0, 0)
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            #dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT # board
            #dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE # ball
            board_dof_props['stiffness'][:] = 0.0
            board_dof_props['damping'][:] = 0.0

            self.envs.append(env_ptr)
            self.board_handles.append(board_handle)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)

        self.board_indices = to_torch(self.board_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        # retrieve environment observations from buffer
        board_pos = self.obs_buf[:, 0]
        board_vel = self.obs_buf[:, 1]
        ball_pos = self.obs_buf[:, 2:5]
        force_value = self.obs_buf[:, 5:11]
        success_count = self.success_count

        self.rew_buf[:], self.reset_buf[:], self.success_count = compute_board_dribble_reward(
            board_pos,
            board_vel,
            ball_pos,
            actions,
            success_count,
            self.reset_dist,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        # print(self.obs_buf)
        # print(self.board_dof_pos)
        # print(self.board_dof_vel)

        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
          
        # self.obs_buf[env_ids, 0] = self.board_dof_pos[env_ids, 0].squeeze()
        # self.obs_buf[env_ids, 1] = self.board_dof_vel[env_ids, 0].squeeze()
        # self.obs_buf[env_ids, 2:5] = self.object_pos[env_ids, 0:3].squeeze()

        self.obs_buf[:, 0] = self.board_dof_pos[:, 0].squeeze()
        self.obs_buf[:, 1] = self.board_dof_vel[:, 0].squeeze()
        self.obs_buf[:, 2:5] = self.object_pos[:, 0:3].squeeze()

        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        sensors_per_env = 1

        force_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        # print("force_sensor_tensor", force_sensor_tensor) 
        
        self.obs_buf[:, 5:11] = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        
        # print("force sensor value-----------",  self.obs_buf[:, 5:11])
       
        return self.obs_buf

    def reset_idx(self, env_ids):
        board_positions = 0.2 * (torch.rand((len(env_ids), self.num_board_dofs), device=self.device) - 0.5)
        board_velocities = 0.5 * (torch.rand((len(env_ids), self.num_board_dofs), device=self.device) - torch.rand((len(env_ids), self.num_board_dofs), device=self.device))

        # ball_posistions = ball_positions + torch.tensor([[0, 0, 0.5]])
        board_velocities = 5.0 * torch.ones((len(env_ids), self.num_board_dofs), device=self.device)

        self.board_dof_pos[env_ids, :] = board_positions[:]
        self.board_dof_vel[env_ids, :] = board_velocities[:]
       
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32 * 2), len(env_ids_int32))
        # env_ids
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        actions_tensor = torch.zeros(self.num_envs * self.num_board_dofs, device=self.device, dtype=torch.float) # 1 × (環境数×board_dof)
        print("actions_tensor", actions_tensor)
        actions_tensor[::self.num_board_dofs] = actions.to(self.device).squeeze() * self.max_push_effort
        print("actions_tensor", actions_tensor)
        
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

        print("hit time?", self.gym.get_sim_time(self.sim))
        print("hit time?", self.gym.get_sim_time(self.sim))

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
        board_pos,
        board_vel,
        ball_pos,
        actions,
        success_count,
        reset_dist,
        reset_buf,
        progress_buf,
        max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, int, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, int]

    print("success_count", success_count)
    board2ball_dist = ball_pos[0, 2] - board_pos
    # board2ball_dist = ball_pos[0, 2]
    #print("ball_pos[0, 2]", ball_pos[0, 2])
    #print("board_pos", board_pos)
    # action_cost = 0.5 * 1.0 * abs(board_vel)**2
    # action_cost = 5 * abs(board_vel)**2 
    #print("board-ball distance =", board2ball_dist)
    #print("size", board2ball_dist.size())

    # energy penalty for movement
    print("actions", actions)
    action_cost = 0.3*torch.sum(actions ** 2, dim=-1)
    success_reward = torch.ones_like(action_cost)
    
    reward = board2ball_dist - action_cost + success_count * success_reward
    # reward = (board2ball_dist - action_cost)
    print("reward------", reward)

    # adjust reward for reset agents
    # torch.where(condition, x, y) contidion = true -> x, condition = false -> y
    # reset_dist = 2.0

    reward = torch.where(torch.abs(board2ball_dist) < reset_dist, torch.ones_like(reward) * -2.0, reward)
    
    # reward = torch.where(torch.abs(board2ball_dist) > 2.0, reward, torch.zeros_like(reward))
    if (board2ball_dist > 3).all().item():
         success_count += 1
    else:
         success_count = 0
         reward = -reward

    # reset
    if success_count > 5:
        reset = torch.ones_like(reset_buf)
        success_count = 0
    else:
        reset = reset_buf
     
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset, success_count
