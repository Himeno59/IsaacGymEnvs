# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy 
from typing import Dict, Any, Tuple, List, Set

import gym
from gym import spaces #環境の観測空間や行動空間の定義に仕様

from isaacgym import gymtorch, gymapi
#gymtorch:IsaacGym環境をPytorchのテンソルとして扱うユーティリティ関数を提供
#gymapi:IsaacGymへのAPIへのアクセス

from isaacgym.torch_utils import to_torch
from isaacgymenvs.utils.dr_utils import get_property_setter_map, get_property_getter_map, \
    get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

import torch #PyTorch
import numpy as np
import operator, random
from copy import deepcopy
from isaacgymenvs.utils.utils import nested_dict_get_attr, nested_dict_set_attr

from collections import deque

import sys #システムに関連するパラメータと関数

import abc #Abstaract Base Classes 抽象基底クラス作成のためのモジュール
from abc import ABC #継承を利用して抽象基底クラスを定義できる

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

def _create_sim_once(gym, *args, **kwargs): #create_simの中で呼ばれる
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs) #gymはimportしてきたクラスみたいなもの
        return EXISTING_SIM

#抽象基底クラス
#関数の宣言だけされていて、これを継承して具象クラスを作成し、メソッドをオーバーライドしていく
class Env(ABC):
    def __init__(self, config: Dict[str, Any], rl_device: str, sim_device: str, graphics_device_id: int, headless: bool): 
        """Initialise the env.

        Args:
            config: the configuration dictionary. 環境の設定情報の保持
　　　　　　rl_evice: 強化学習のデバイス cpu or gpu の指定
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu' 物理シミュレーションのデバイス
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.　GUiなしかどうか
        """

        #sim_deviceの初期化
        split_device = sim_device.split(":") #":"で区切って返す
        self.device_type = split_device[0] #前半がデバイスのタイプ cudaとか
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0 #後半がid,もしidがない->デバイスがcpuとかなら0を返す

        #sim_deviceがGPUデバイスをしていない場合にself.deviceを"cpu"に設定する
        self.device = "cpu" 
        if config["sim"]["use_gpu_pipeline"]: #GPUパイプラインを使用する指定の場合
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False #cpuパイプラインに切り替え


        #強化学習のデバイスの初期化
        self.rl_device = rl_device

        # Rendering
        # if training in a headless mode
        self.headless = headless

        #カメラセンサーを有効にするかどうか
        #カメラセンサーが無効化されていてかつヘッドレスモードの場合、描画デバイスのIDとして-1が使用される
        #それ以外の場合は引数で与えられたgraphics_device_idを入れる
        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        #configの情報から環境の情報を登録していく
        self.num_environments = config["env"]["numEnvs"] #環境の数
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments　エージェント数

        self.num_observations = config["env"].get("numObservations", 0) #観測空間の次元数
        self.num_states = config["env"].get("numStates", 0) #状態空間の次元数

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf) #観測空間の定義
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf) #状態空間の定義
        #spaces.Box:連続地を持つボックス型の空間を表現するためのクラス
        #-∞から∞に設定されること、制約されない連続値の空間が定義される

        self.num_actions = config["env"]["numActions"] #アクションの数<-次元数ではない？
        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1) #制御の頻度を表す逆数

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
　　　　#環境のアクション空間の定義
        
        self.clip_obs = config["env"].get("clipObservations", np.Inf) #観測値のクリッピング範囲
        self.clip_actions = config["env"].get("clipActions", np.Inf) #アクション値のクリッピング範囲

        # Total number of training frames since the beginning of the experiment.
        # We get this information from the learning algorithm rather than tracking ourselves.
        # The learning algorithm tracks the total number of frames since the beginning of training and accounts for
        # experiments restart/resumes. This means this number can be > 0 right after initialization if we resume the
        # experiment.
        self.total_train_env_frames = 0
        #実験開始からの総訓練フレーム数
        #学習アルゴリズムから取得
        #実験の再開も考慮に入れるため、初期化した後でも0より大きい場合がある

    @abc.abstractmethod 
    def allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""
        #観測値、報酬、行動、終了情報などのためのTorchのバッファを作成

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.
        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        #環境の物理的なステップを進める
        #適用するアクションを入れて、観測値、報酬、リセット情報、追加の情報のタプルを返す
        #観測値は辞書型

    @abc.abstractmethod
    def reset(self)-> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        #環境をリセットする
        #観測値の辞書を返す

    @abc.abstractmethod
    def reset_idx(self, env_ids: torch.Tensor):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """
        #指定されたインデックスを持つ環境をリセットする

    #以下はget関数
    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space
    #環境の観測空間の取得

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space
    #環境の行動空間を取得

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments
    #環境の数を取得

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.num_actions
    #環境のアクションの次元数を獲得

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.num_observations
    #環境の観測の次元数を獲得


#抽象基底クラスであるEnvを継承したクラス
#Env->VecTask->それぞれのタスク という感じで継承してそう
class VecTask(Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False): 
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. Trueにしておくとこのenv.renderでrgb配列を取得できる
            force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True) Trueにしておくとステップごとにレンダリングをすることを矯正する
        """
        # super().__init__(config, rl_device, sim_device, graphics_device_id, headless, use_dict_obs)
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)
        #Envの方で定義していなかった変数の初期化
        self.virtual_screen_capture = virtual_screen_capture
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()
        self.force_render = force_render

        #使用する物理エンジンの設定
        self.sim_params = self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        # optimization flags for pytorch JIT(just in time)
        # 最適化フラグ
        torch._C._jit_set_profiling_mode(False)
        # JITプロファイリング：実行時にコードのプロファイリング情報を収集し、最適化のためのヒントを提供する機能、今回は無効にしている
        torch._C._jit_set_profiling_executor(False)
        #JITプロファイリングの実行モードを無効にする。実行時に最適化を行う
        #これらは特定の実行環境や使用方法に置いてJITの影響を受けたくない場合に使用されることがある
        #理由はよくわからないが、実行速度の向上やリソースの節約が期待できるとか、実行結果を再現性野あるものにしたい場合にオフにする

        #gym apiはこのオブジェクトから取得できるように成る
        self.gym = gymapi.acquire_gym()

        
        self.first_randomization = True
        self.original_props = {} #環境の元のプロパティを保持する辞書型
        self.dr_randomizations = {} #dr：domain random　ランダム化するものを入れておく？
        self.actor_params_generator = None #アクターパラメータ生成器
        self.extern_actor_params = {} #外部アクターパラメータ??
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.sim_initialized = False
        self.create_sim() #シミュレーションの初期化と生成？色んなバッファの初期化がされている
        self.gym.prepare_sim(self.sim) #シミュレーションのための内部データ構造の初期化
        self.sim_initialized = True

        self.set_viewer() #ビューワの初期化
        self.allocate_buffers()

        self.obs_dict = {}

    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT") #escでストップ
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync") #Vでレンダリングの有無を変更

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def allocate_buffers(self): #バッファの割当　事前に必要なメモリ領域を確保するみたいな感じ？
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """
        #Envクラスを継承したクラスにおいて、観測と状態を設定するために使用され、stepや関連する他の関数で読み取られる

        # allocate buffers
        # PyTorchでいい感じに割り当ててる？
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def get_state(self):
        """Returns the state buffer of the environment (the privileged observations for asymmetric training)."""
        return torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
    #torch.clampという関数はテンソルの値を指定された範囲内に制約するために使用される

    @abc.abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """
        #環境に対してアクション(ex.トルク、位置目標など)を適用するために使う
        #派生クラスでアクションを環境に反映される処理を記述する必要がある

    @abc.abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""
        #報酬と観測値を計算し、必要な場合にリセットするために使用される
        #stepの後に呼ばれる
        #派生クラスで報酬と観測値の計算、環境音リセット処理を記述する必要がある

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        #一番重要そうなところ
        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim) #gym.simulateで物理ステップを実行

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        # actor-criticは価値観数ベースと方策勾配法ベースの考え方を組み合わせた手法
        # assymmetricとは？非対称？
        # エージェントの行動方策と価値観数が異なるネットワークまたはパラメータを持つことを意味する
        # 非対称の設定では行動方策は主にエージェントの行動を決定する役割を果たし、価値観数は環境の状態や行動の価値を評価する役割を果たす
        if self.num_states > 0: #状態空間の次元数が0より大きければ
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras #観測値(obs_dict)、報酬、リセットバッファ、その他の情報が返される
    #観測値は辞書、報酬とリセットバッファはPyTorchテンソルとして返される

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        #アクションの初期化
        
        actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)

        return actions

    def reset_idx(self, env_idx):
        """Reset environment with indces in env_idx. 
        Should be implemented in an environment class inherited from VecTask.
        """
        #継承先のクラスで実装する必要あり
        pass

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        #最初の観測の際に一度だけ呼び出される
        #具体的には自分で実装する必要あり
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def reset_done(self):
        """Reset the environment.
        Returns:
            Observation dictionary, indices of environments being reset
        """
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten() 
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids
    #リセット後の観測辞書とリセットが行われた環境の情報が提供される

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)
                #ポーリング：ビューアからのイベントを継続的に監視し、新しいイベントが発生したどうかを定期的にチェックする処理

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """

        ## parse：解析する
        ## __：クラスの外部から直接アクセスできないようにする、クラス内の別のメソッドや処理の補助として使用されることが期待される
        sim_params = gymapi.SimParams()

        # check correct up-axis：上軸 重力の方向や座標系の上方向を定義するために使用される
        if config_sim["up_axis"] not in ["z", "y"]: #zかyじゃないといけないということ？
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters：パラメーターをシミュレーションに設定
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2) #サブステップとは？

        # assign up-axis zかy
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
                        #setattr：オブジェクトの属性を動的に設定するために使用される
                        #setattr(object attribute, value)
                        #for文で回しながら1つずつ設定していく
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params
    

    """
    Domain Randomization methods
    """
    #ドメインランダム化：機械学習やロボティクスにおいて使用される手法
    #エージェントが学習させる環境を人工的に多様化することで汎化性能やロバスト性を向上させる

    def get_actor_params_info(self, dr_params: Dict[str, Any], env):
        """Generate a flat array of actor params, their names and ranges.

        Returns:
            The array
        """
        #dr：domain random化

        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = [] #下限のリスト
        highs = [] #上限のリスト

        #gymオブジェクトからアクターパラメータを取得する関数を提供
        param_getters_map = get_property_getter_map(self.gym)

        
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            #アクターハンドル：物理シミュレーション環境内で特定のアクターオブジェクトを一意に識別するための識別子
            #これを介して情報にアクセスしたりする

            #プロパティは属性へのアクセス方法を制御
            #アトリビュートはオブジェクトの特性やデータそのものを表す
            #prop：property
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props] #リスト化する
                for prop_idx, prop in enumerate(props):
                    #attr：attribute
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name+'_' + str(prop_idx) + '_'+attr
                        lo_hi = attr_randomization_params['range'] #low-high
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0*float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray): #プロパティがnumpy配列の場合
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name+'_'+str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else: #そうでない場合
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs #生成したパラメータを返す
    #この関数はドメインランダム化のために使用されるアクターパラメータの情報を抽出するためのユーティリティ関数


    #最後の関数
    #ランダム化
    def apply_randomizations(self, dr_params):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
        """

        # If we don't have a randomization frequency, randomize every step
        # デフォルトでは1回のステップごとにrandomizationが適用
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        
        #最初の呼び出し時は全ての要素に対して適用
        #非環境パラメータ：環境内の物理的な要素やオブジェクト以外のパラメータ、観測データの前処理やエージェントの行動の制御に関連んするパラメータなど
         #最後のrandomizationからrand_freqステップ以上経過している場合に適用
        #物理環境：環境の物理的なパラメータ
         #randomization頻度の閾値を超えたリセットバッファ内の物理環境に対して適用
        self.last_step = self.gym.get_frame_count(self.sim)

        #非環境パラメータのランダム化 nonenv
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs)) #全ての環境に対して非環境パラメータのランダム化
        else: #rand_frewステップ崇敬化したらフラグを立てる
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            #random_bufがrand_freq以上の値を持つ環境を特定
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            #リセットが必要な環境(reset_buf)との論理積を取る
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            #ランダマイゼーションが必要な環境のリストを取得
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0 #ランダマイゼーションが実行された環境において値をリセット

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step #last_rand_stepを現在のステップ数(last_step)に更新
            

        param_setters_map = get_property_setter_map(self.gym) #プロパティのセッターのマップ
        param_setter_defaults_map = get_default_setter_args(self.gym) 
        param_getters_map = get_property_getter_map(self.gym)　#プロパティのゲッターのマップ
        #マップ：プロパティの名前をキーとしてそれに対応するセッターとゲッターの関数を値として持つ辞書
        #このマップを使用することでプロパティの名前を指定して対応するセッターとゲッターの関数を取得できる

        # On first iteration, check the number of buckets
        #最初の反復時にバケットの数をチェックするための処理
        #バケット：一定の範囲内に値を分割してグループ化するためのデータ構造
        #0~100までの数値を10個のバケットに分割->バケット1:0-9, バケット2:10-19みたいな感じ
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        #非物理的なパラメータ(observationとactions)に対してドメインランダマイゼーションを適用するための処理
        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"] #確率分布の指定(gausian or uniform：一様分布 )
                op_type = dr_params[nonphysical_param]["operation"] #演算タイプの指定(additive or scaling)
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None #スケジュールタイプ?
                #パラメタ変化を制御する方法：線形や定数などがある
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None #スケジュールステップ
                #上のスケジュールの制御に使用される具体的なステップ数を表し、タイプによって異なる
                #線形ならステップ数に基づいてパラメタの変化がスケジューリングされる
                #定数ならこのステップ数を超えるかどうかで制御される

                #演算子の指定
                op = operator.add if op_type == 'additive' else operator.mul

                
                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                #dist:distribution 分布   
                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"] #mu:平均<-ミュー(μ) var:分散
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.]) #相関correlated

                    if op_type == 'additive': #加算
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling': #乗算
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param): #ランダマイゼーションされたノイズをテンソルに加えるために使用
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None: #相関がなければランダムなテンソルを生成して保持
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr'] #相関ノイズcorrをvar_corrとmu_corrでスケーリング 
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])
                    #op:operandと用いて元のテンソルに正規分布のノイズを加える

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda} #ランダマーゼーションのパラメータの設定

                elif dist == 'uniform': #一様分布の場合　流れはガウス分布のときと同じ
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize: #もしdr_paramsのなかにsim_paramsが存在し、かつnonenvパラメタのランダマイせーションがtrueの場合
            prop_attrs = dr_params["sim_params"] #sim_paramsに色々含まれている
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop) #シミュレーションに設定

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.

        #ランダマイゼーションフレームワークにおいて、デフォルトでは各シミュレーションパラメータを独立にランダム化するが、特定の環境などに置いてはパラメタ間の相互作用や共分散を考慮するいhつようがあるかもしれない
        #self.actor_params_generatorとカスタムな分布製世紀として設定し、任意の分布からアクターシミュレーションパラメータをサンプリングすることができる
        #要はより柔軟なアクターシミュレーションパラメータのランダム化が可能になる
        #ここよくわからん
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

                
        # randomise all attributes of each actor (hand, cube etc..)
        # actor_properties are (stiffness, damping etc..)

        # Loop over actors, then loop over envs, then loop over their props 
        # and lastly loop over the ranges of the params 

        #アクタープロパティのランダム化を実行するためのループと制御フロー
        for actor, actor_properties in dr_params["actor_params"].items():

            # Loop over all envs as this part is not tensorised yet 
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor) #handleの取得
                extern_sample = self.extern_actor_params[env_id] #exter:external 外部の

                #ランダム化の適用
                # randomise dof_props, rigid_body, rigid_shape properties 
                # all obtained from the YAML file
                # EXAMPLE: prop name: dof_properties, rigid_body_properties, rigid_shape properties  
                #          prop_attrs: 
                #               {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform'}
                #               {'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform'}
                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue

                    if prop_name == 'scale':
                        setup_only = prop_attrs.get('setup_only', False)
                        if (setup_only and not self.sim_initialized) or not setup_only:
                            attr_randomization_params = prop_attrs
                            sample = generate_random_samples(attr_randomization_params, 1,
                                                             self.last_step, None)
                            og_scale = 1
                            if attr_randomization_params['operation'] == 'scaling':
                                new_scale = og_scale * sample
                            elif attr_randomization_params['operation'] == 'additive':
                                new_scale = og_scale + sample
                            self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    set_random_properties = True

                    if isinstance(prop, list): #リストかリストでないかで場合分け
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]): #ループで各要素に対してプロパティの値を変更するための処理が行われる
                            for attr, attr_randomization_params in prop_attrs.items():
                                setup_only = attr_randomization_params.get('setup_only', False)
                                if (setup_only and not self.sim_initialized) or not setup_only:
                                    smpl = None
                                    if self.actor_params_generator is not None:
                                        smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                            extern_sample, extern_offsets[env_id], p, attr)
                                    apply_random_samples(
                                        p, og_p, attr, attr_randomization_params,
                                        self.last_step, smpl) #ランダム化の適用
                                else:
                                    set_random_properties = False
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            setup_only = attr_randomization_params.get('setup_only', False)
                            if (setup_only and not self.sim_initialized) or not setup_only:
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], prop, attr)
                                apply_random_samples(
                                    prop, self.original_props[prop_name], attr,
                                    attr_randomization_params, self.last_step, smpl)
                            else:
                                set_random_properties = False

                    if set_random_properties:
                        setter = param_setters_map[prop_name]
                        default_args = param_setter_defaults_map[prop_name]
                        setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample　サイズ(次元)の確認
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id,
                              'extern_offset', extern_offsets[env_id],
                              'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False
