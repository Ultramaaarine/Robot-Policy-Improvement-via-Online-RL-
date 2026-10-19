```python
# Task-specific CALVIN simulation environment for diffusion-policy rollout.
# It wraps the original PlayTableSimEnv to configure skill-dependent scene states,
# expose policy-friendly observations, compute drawer-task rewards and success,
# and execute Cartesian robot actions in PyBullet.

import logging
import hydra
import gc
import sys
import os
import pkgutil
import gym
import pybullet as p
import pybullet_utils.bullet_client as bc
import numpy as np
import math
from calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv

logger = logging.getLogger(__name__)

# here are the indices of data in the observation vector
# here [0,1,2] - gripper position; [3,4,5] - gripper orientation, [14] - gripper action, [6] - gripper width

# Indices used to extract policy-relevant robot states from CALVIN's robot observation vector.
CALVIN_POSITION_INDICES = np.array([0, 1, 2])
CALVIN_ORIENTATION_INDICES = np.array([3, 4, 5])
CALVIN_JOINTS_INDICES = np.array([7, 8, 9, 10, 11, 12, 13])
CALVIN_6D_INDICES = np.concatenate([CALVIN_POSITION_INDICES, CALVIN_ORIENTATION_INDICES])

from gym.envs.registration import register

# Register the task-specific CALVIN environment so it can be created through Gym.
register(id="skill-env", entry_point="sac_diffusion.envs.calvin:CalvinSkillEnv", max_episode_steps=64)


class CalvinSkillEnv(PlayTableSimEnv):
    def __init__(self, cfg, skill, init_pos):
        print("===Initializing CalvinSkillEnv ===")

        # Store the current CALVIN task, such as open_drawer or close_drawer.
        self.skill = skill

        # Remove Hydra-specific fields before passing the environment configuration
        # into the original CALVIN PlayTableSimEnv.
        pt_cfg = {**cfg.calvin_env.env}
        pt_cfg.pop("_target_", None)
        pt_cfg.pop("_recursive_", None)

        # Configure task-dependent scene initialization before the parent environment is created.
        self.set_scene_cfg(pt_cfg["scene_cfg"])
        
        # Save robot and camera configurations for later simulator loading.
        self.robot_cfg = pt_cfg["robot_cfg"]
        self.cameras_c = pt_cfg["cameras"]

        # Initialize the original CALVIN simulation environment.
        super(CalvinSkillEnv, self).__init__(**pt_cfg) # still env config

        # Save the initial robot base and end-effector pose for later reset/randomization.
        self.init_base_pos, self.init_base_orn = self.p.getBasePositionAndOrientation(self.robot.robot_uid)
        self.init_gripper_pos = self.robot.target_pos
        self.init_gripper_orn = self.robot.target_orn

        # Gaussian noise used to randomize the initial end-effector position.
        self.ee_noise = np.array(cfg.ee_noise)

        # Select which observation modalities should be exposed to the policy.
        self.obs_space = cfg.obs_space

        # Optional manually specified initial end-effector position.
        self.init_pos = init_pos

        # Store task-specific episode length and CALVIN task definitions.
        self.max_episode_steps = self.skill.max_steps
        self.tasks = hydra.utils.instantiate(cfg.calvin_env.tasks)

        self.is_source = cfg.is_source

        # Ground-truth interaction keypoint used as rollout reach target.
        self.gt_keypoint = None

        print("===CalvinSkillEnv is initialized===")


    def load(self):

        # Rebuild the PyBullet simulation from scratch.
        logger.info("Resetting simulation")
        self.p.resetSimulation(physicsClientId=self.cid)

        logger.info("Setting gravity")
        self.p.setGravity(0, 0, -9.8, physicsClientId=self.cid)

        # Instantiate robot and scene from Hydra configuration.
        self.robot = hydra.utils.instantiate(self.robot_cfg, cid=self.cid)
        self.scene = hydra.utils.instantiate(
            self.scene_cfg,
            p=self.p,
            cid=self.cid,
            np_random=self.np_random
        )

        # Load robot and scene objects into PyBullet.
        self.robot.load()
        self.scene.load()

        # init cameras after scene is loaded to have robot id available
        # Cameras are initialized after robot/scene loading because they require
        # the robot ID and scene object references.
        self.cameras = [
            hydra.utils.instantiate(
                self.cameras_c[name],
                cid=self.cid,
                robot_id=self.robot.robot_uid,
                objects=self.scene.get_objects()
            )
            for name in self.cameras_c
        ]

        #print(f"calvin skill env initialized") no respose, calvin skill is not instantialized


    def initialize_bullet(self, bullet_time_step, render_width, render_height):

        # Store physics simulation rate and rendering resolution.
        self.bts = bullet_time_step
        self.rhw = (render_height, render_width)

        # Create a new PyBullet client only when no active connection exists.
        if not self.isConnected():
            self.ownsPhysicsClient = True

            # Connect to shared-memory Bullet server for VR mode.
            if self.use_vr:
                self.p = bc.BulletClient(connection_mode=p.SHARED_MEMORY)
                cid = self.p._client

                if cid < 0:
                    logger.error(
                        "Failed to connect to SHARED_MEMORY bullet server.\n"
                        " Is it running?"
                    )
                    sys.exit(1)

                self.p.setRealTimeSimulation(
                    enableRealTimeSimulation=1,
                    physicsClientId=cid
                )

            # Start PyBullet with a graphical interface.
            elif self.show_gui:
                self.p = bc.BulletClient(connection_mode=p.GUI)
                cid = self.p._client

                if cid < 0:
                    logger.error("Failed to connect to GUI.")

            # Use EGL for headless GPU rendering.
            elif self.use_egl:
                options = f"--width={self.rhw[1]} --height={self.rhw[0]}"
                self.p = bc.BulletClient(p.DIRECT, options=options)

                self.p.setPhysicsEngineParameter(
                    deterministicOverlappingPairs=1
                )

                cid = self.p._client

                # Disable unnecessary PyBullet debug visualization windows.
                self.p.configureDebugVisualizer(
                    p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                    0,
                    physicsClientId=cid
                )
                self.p.configureDebugVisualizer(
                    p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                    0,
                    physicsClientId=cid
                )
                self.p.configureDebugVisualizer(
                    p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                    0,
                    physicsClientId=cid
                )

                # Load EGL plugin for headless GPU rendering.
                egl = pkgutil.get_loader("eglRenderer")
                logger.info(
                    "Loading EGL plugin (may segfault on misconfigured systems)..."
                )

                if egl:
                    plugin = self.p.loadPlugin(
                        egl.get_filename(),
                        "_eglRendererPlugin"
                    )
                else:
                    plugin = self.p.loadPlugin("eglRendererPlugin")

                if plugin < 0:
                    logger.error("\nPlugin Failed to load!\n")
                    sys.exit()

                # set environment variable for tacto renderer
                os.environ["PYOPENGL_PLATFORM"] = "egl"

                logger.info("Successfully loaded egl plugin")

            # Standard headless CPU rendering / simulation mode.
            else:
                self.p = bc.BulletClient(p.DIRECT)
                cid = self.p._client

                if cid < 0:
                    logger.error("Failed to start DIRECT bullet mode.")

            logger.info(f"Connected to server with id: {cid}")

            # Store and configure the active physics client.
            self.cid = cid

            self.p.resetSimulation(
                physicsClientId=self.cid
            )

            self.p.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self.cid
            )

            self.p.configureDebugVisualizer(
                self.p.COV_ENABLE_GUI,
                0,
                physicsClientId=self.cid
            )

            logger.info(f"Connected to server with id: {self.cid}")

            # Set simulation timestep from Bullet update frequency.
            self.p.setTimeStep(
                1.0 / self.bts,
                physicsClientId=self.cid
            )

            return cid


    def close(self):

        # Disconnect only if this environment created the Bullet client.
        if self.ownsPhysicsClient:
            print("disconnecting id %d from server" % self.cid)

            if self.cid >= 0 and self.p is not None:
                print("CID: " + str(self.cid))

                try:
                    self.p.disconnect(
                        physicsClientId=self.cid
                    )
                except:
                    pass

        else:
            print("does not own physics client id")


    def isConnected(self):

        # Check whether the current PyBullet connection is still active.
        connected = False

        if self.p is not None:
            connected = self.p.getConnectionInfo()["isConnected"] == 1

            # Clean up stale Bullet clients.
            if not connected:
                try:
                    self.p.disconnect(self.cid)
                    del p
                except:
                    pass

                gc.collect()

        return connected


    def object_by_name(self, name):

        # Search scene objects by task-relevant name/effect.
        doors = [o for o in self.scene.doors if o.name == name]
        buttons = [o for o in self.scene.buttons if o.effect == name]

        if len(doors) != 0:
            return doors[0]

        elif len(buttons) != 0:
            return buttons[0]


    def object_position(self, obj):

        # Find the PyBullet link associated with the requested scene object.
        base = self.scene.fixed_objects[0]

        ln = self.p.getJointInfo(
            base.uid,
            obj.joint_index,
            physicsClientId=self.cid
        )[12].decode()

        # Return the object's world-frame position.
        return self.p.getLinkState(
            base.uid,
            base.get_info()["links"][ln],
            physicsClientId=self.cid
        )[0]


    def task_object_position(self):

        # Extract the task name without the "calvin_" prefix.
        task_filter = self.skill.name.split("_", 1)[1]

        # Find the object associated with the selected CALVIN task.
        object = self.object_by_name(
            self.tasks.tasks[task_filter].args[0]
        )

        # Shift the physical object position to a useful manipulation keypoint.
        if object.name == "base__drawer":
            offset = np.array([0.0, -0.17, 0.03])

        else:
            raise NotImplementedError

        return self.object_position(object) + offset


    def set_scene_cfg(self, scene_cfg):

        # Initialize the drawer according to the task:
        # close_drawer starts open, while open_drawer starts closed.
        if self.skill.name == "calvin_close_drawer":
            scene_cfg.objects.fixed_objects.table.joints.base__drawer.initial_state = 0.18

        elif self.skill.name == "calvin_open_drawer":
            scene_cfg.objects.fixed_objects.table.joints.base__drawer.initial_state = 0.0

        self.scene_cfg = scene_cfg


    def set_gt_keypoint(self):

        # Store the task-specific ground-truth interaction target.
        self.gt_keypoint = self.task_object_position()


    def sample_base_shift(self):

        #shift = np.array(self.skill.T_obj_gripper)

        # Sample Gaussian displacement for randomized initialization.
        pos_shift =  np.random.normal(0.0, self.ee_noise, 3) #+ shift 

        gripper_pos =   pos_shift #+self.task_object_position()

        delta_pos = gripper_pos - self.init_gripper_pos

        bp, born = self.init_base_pos, self.init_base_orn

        # Apply the sampled end-effector displacement to the robot base position.
        bp = bp + delta_pos

        return bp, born


    def sample_ee_pose(self):

        # Choose either the current robot target position or a manually provided start position.
        if self.init_pos is None:
            self.init_gripper_pos = self.robot.target_pos
        else:
            self.init_gripper_pos = self.init_pos

        self.init_gripper_orn = self.robot.target_orn

        # offset = np.random.uniform(-self.ee_noise, self.ee_noise, 3)

        # Randomize rollout start state with Gaussian Cartesian noise.
        offset = np.random.normal(0.0, self.ee_noise, 3)

        gripper_pos = self.init_gripper_pos + offset

        # Keep the initial orientation unchanged.
        gripper_orn = self.init_gripper_orn

        return gripper_pos, gripper_orn


    def calibrate_EE_start_state(
        self,
        error_margin=0.01,
        max_checks=15,
        desired_pos=None
    ):
        """
        Samples a random but good starting point and moves the end effector to that point
        """

        # Sample a randomized start pose unless a desired position is explicitly provided.
        if desired_pos is None:
            ee_pos, ee_orn = self.sample_ee_pose()

        else:
            ee_pos = desired_pos
            ee_orn = self.robot.target_orn

        count = 0

        # Use absolute Cartesian control to move the robot to the start pose.
        action = {}
        action["type"] = "cartesian_abs"
        action["action"] = np.concatenate(
            [ee_pos, ee_orn, [-1]],
            axis=0
        )

        curr_pos = self.get_obs()["position"]

        # Repeatedly command the desired pose until the end effector reaches it.
        while np.linalg.norm(curr_pos - ee_pos) > error_margin:
            obs, _, _, _ = self.step(action)
            curr_pos = obs["position"]

            if count >= max_checks:
                print("NOT REACHING")
                return False

            count += 1

        # Synchronize the robot controller target with the reached pose.
        self.robot.update_target_pose()

        return True


    def reset(self, robot_obs=None, scene_obs=None):

        # Reinitialize Bullet and reload the scene if the simulator is disconnected.
        if not self.isConnected():
            h, w = self.rhw
            self.initialize_bullet(self.bts, w, h)
            self.load()

        # Reset robot and scene to the requested initial state.
        self.robot.reset(robot_obs)
        self.scene.reset(scene_obs)

        if robot_obs is None:
            pass

            #npos, norn = self.sample_base_shift()
            #self.p.resetBasePositionAndOrientation(self.robot.robot_uid, npos, norn)

        #self.robot.reset(robot_obs)

        # Advance one physics step after reset.
        self.p.stepSimulation(
            physicsClientId=self.cid
        )
        
        self.scene.reset(scene_obs)

        # Store the initial task state for later success evaluation.
        self.start_info = self.get_info()

        # Compute the environment-provided rollout target.
        self.set_gt_keypoint()

        return self.get_obs()


    # @staticmethod
    def get_action_space(self):
        """End effector position and gripper width relative displacement"""

        # CALVIN uses a normalized 7D Cartesian + gripper action.
        return gym.spaces.Box(
            np.array([-1] * 7),
            np.array([1] * 7)
        )


    # @staticmethod
    @property
    def get_observation_space(self):
        """Return only position and gripper_width by default"""

        # Construct the observation space from enabled modalities.
        observation_space = {}

        if "pos" in self.obs_space:
            observation_space["position"] = gym.spaces.Box(
                low=-1,
                high=1,
                shape=[3]
            )

        if "orn" in self.obs_space:
            observation_space["orientation"] = gym.spaces.Box(
                low=-1,
                high=1,
                shape=[3]
            )

        if "joints" in self.obs_space:
            observation_space["joints"] = gym.spaces.Box(
                low=-math.pi,
                high=math.pi,
                shape=[7]
            )

        if "gripper" in self.obs_space:
            observation_space["rgb_gripper"] = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(3, 84, 84)
            )

        if "state" in self.obs_space:
            observation_space["state"] = gym.spaces.Box(
                low=-1,
                high=1,
                shape=[21]
            )

        return gym.spaces.Dict(observation_space)


    def get_obs(self):

        # Obtain the original full CALVIN observation.
        obs = super(CalvinSkillEnv, self).get_obs()

        #print(f"obs in skill env has keys:{obs.keys()}")
        #print(f"rgb_obs has keys: {obs["rgb_obs"].keys()}")
        #print(f"scene_obs are: {obs["scene_obs"]}")

        # Build a compact policy-specific observation dictionary.
        nobs = {}

        robs = np.array(obs["robot_obs"])

        # Extract Cartesian end-effector position.
        if "pos" in self.obs_space:
            nobs["position"] = robs[CALVIN_POSITION_INDICES]

        # Extract Euler end-effector orientation.
        if "orn" in self.obs_space:
            nobs["orientation"] = robs[CALVIN_ORIENTATION_INDICES]

        # Extract seven robot joint positions.
        if "joints" in self.obs_space:
            nobs["joints"] = robs[CALVIN_JOINTS_INDICES]

        if "gripper" in self.obs_space:
            gr_rgb = obs["rgb_obs"]["rgb_gripper"]

            # Convert image layout from HWC to CHW for the visual encoder.
            nobs["rgb_gripper"] = np.moveaxis(gr_rgb, 2, 0)

            cam = self.cam_by_name("gripper")

            # Camera view/extrinsic matrix.
            nobs["view_mtx"] = np.array(cam.view_matrix)

            # nobs["proj_mtx"] = np.array(cam.projection_matrix)

            # Camera intrinsic matrix used for geometric calculations.
            nobs["intrinsics"] = self.get_gripper_cam_intrinsics()

        if "state" in self.obs_space:

            # Combine robot and scene observations into one compact state vector.
            nobs["state"] = np.concatenate(
                [obs["robot_obs"], obs["scene_obs"]]
            )[:21]

        return nobs


    def get_pos_from_obs(self, obs):

        # Extract end-effector position from the currently enabled observation format.
        if "pos" in self.obs_space:
            curr_pos = obs["position"]

        elif "state" in self.obs_space:
            curr_pos = obs["state"][:3]

        else:
            raise NotImplementedError

        return curr_pos


    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""

        # Compare the current state against the state recorded at reset.
        current_info = self.get_info()

        task_filter = self.skill.name.split("_", 1)[1]

        # Use CALVIN's built-in task logic to detect whether
        # the requested state transition has been completed.
        task_info = self.tasks.get_task_info_for_set(
            self.start_info,
            current_info,
            task_filter
        )

        print(f"task_info:{task_info}")

        return task_filter in task_info


    def _reward(self,obs):
        """Returns the reward function that will be used
        for the RL algorithm"""

        # Drawer joint position is stored in the compact state vector.
        drawer_slide_length = obs["state"][16]
        
        max_drawaer_slide_length = 0.24
        max_reward = 3

        # Opening task: larger drawer displacement gives larger dense reward.
        if self.skill.name == "calvin_open_drawer":
            dense_reward = (
                (drawer_slide_length-0.0)
                /(max_drawaer_slide_length)
                * max_reward
            )

            dense_reward = np.clip(
                dense_reward,
                0.0,
                max_reward
            ).astype(np.float32)

        # Closing task: smaller drawer displacement gives larger dense reward.
        elif self.skill.name == "calvin_close_drawer":
            dense_reward = (
                1
                -(drawer_slide_length-0.0)
                /(max_drawaer_slide_length)
            ) * max_reward

            dense_reward = np.clip(
                dense_reward,
                0.0,
                max_reward
            ).astype(np.float32)

        # Sparse success reward is computed but currently not added to total reward.
        sparse_reward = int(self._success()) * 10

        # Current implementation uses only dense drawer-state reward.
        total_reward = dense_reward#+sparse_reward

        r_info = {"reward": total_reward}

        return total_reward, r_info


    def _termination(self):
        """Indicates if the robot has reached a terminal state"""

        # Terminate the rollout immediately when the CALVIN task succeeds.
        success = self._success()
        done = success

        d_info = {"success": success}

        print(f"done:{done}")

        return done, d_info


    def step(self, action):
        """Performing a relative action in the environment
        input:
            action: 7 tuple containing
                    Position x, y, z.
                    Angle in rad x, y, z.
                    Gripper action
                    each value in range (-1, 1)
        output:
            observation, reward, done info
        """

        print("use calvinskill env for rollout")

        # Expand position-only policy output into CALVIN's 7D
        # Cartesian + orientation + gripper action format.
        if len(action) == 3:
            print("using env wrapper to build env")

            coords = action

            # Keep orientation command at zero for position-only policies.
            orientation = np.zeros(3)

            # Use a fixed gripper command.
            gripper_action = [1]

            env_action = np.concatenate(
                [coords, orientation, gripper_action],
                axis=0
            )

        else:
            # Full policy action is passed directly to the robot.
            env_action = action

        # Send policy action to the robot controller.
        self.robot.apply_action(env_action)

        # One policy action corresponds to several internal physics steps.
        for i in range(self.action_repeat):
            self.p.stepSimulation(
                physicsClientId=self.cid
            )

        # Update scene dynamics after robot motion.
        self.scene.step()

        # Read the new environment state.
        obs = self.get_obs()

        #print(f"in step, obs has keys: {obs.keys()}")

        print(f"drawer open length: {obs["state"][16]}")

        # Collect environment metadata.
        info = self.get_info()

        # Compute reward and terminal state after the action.
        reward, r_info = self._reward(obs)
        done, d_info = self._termination()

        info.update(r_info)
        info.update(d_info)
        
        # Query CALVIN task state for debugging/monitoring.
        task_filter = self.skill.name.split("_", 1)[1]

        task_info = self.tasks.get_task_info_for_set(
            self.start_info,
            info,
            task_filter
        )

        print(f"task_info:{task_info}")

        return obs, reward, done, info


    def cam_by_name(self, cam_name):

        # Retrieve the configured camera object by name.
        return [c for c in self.cameras if c.name == cam_name][0]


    def get_gripper_cam_intrinsics(self):

        # Build a pinhole-camera intrinsic matrix from field of view and image size.
        fov = 75
        height = 84
        center = height // 2

        # Convert vertical field of view into focal length in pixel coordinates.
        foc = height / (2 * np.tan(np.deg2rad(fov) / 2))

        fx = -foc
        fy = foc
        cx = center
        cy = center

        # Camera intrinsic matrix:
        # [fx  0  cx]
        # [ 0 fy  cy]
        # [ 0  0   1]
        intrinsics = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ]
        )

        return intrinsics
```
