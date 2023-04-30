import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2
from util import get_image_to_tensor_balanced, get_mask_to_tensor
from scipy.spatial.transform import Rotation as R
import json

class ThorDataset_L(torch.utils.data.Dataset):
    """
    Dataset from DVR (Niemeyer et al. 2020)
    Provides 3D-R2N2 and NMR renderings
    """

    def __init__(
        self,
        path,
        stage="train",
        list_prefix="softras_",
        image_size=None,
        sub_format="shapenet",
        scale_focal=True,
        max_imgs=100000,
        z_near=.3,
        z_far=6.8,
        frame_interval = 5,
        skip_step=None,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
        :param sub_format shapenet | dtu dataset sub-type.
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()
        self.base_path = path
        assert os.path.exists(self.base_path)

        cats = [x for x in glob.glob(os.path.join(path, "*")) if os.path.isdir(x)]
        self.full_scenes = []
        # if stage == "train":
        #     file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        # elif stage == "val":
        #     file_lists = [os.path.join(x, list_prefix + "val.lst") for x in cats]
        # elif stage == "test":
        #     file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]
        scenes = os.listdir(self.base_path)
        if '.ipynb_checkpoints' in scenes:
            scenes.remove('.ipynb_checkpoints')
        if stage == "train":
            #print(7*int(len(scenes)/10))
            scenes = scenes[:8*int(len(scenes)/10)]
        if stage == "val":
            scenes = scenes = scenes[8*int(len(scenes)/10):9*int(len(scenes)/10)]
        if stage == "test":
            #print(9*int(len(scenes)/10))
            scenes = scenes[9*int(len(scenes)/10):]
        self.full_scenes = [os.path.join(self.base_path, x) for x in scenes]
        self.episode_frames = []
        self.cps = []
        self.horizon = []
        self.rots = []
        for scene in self.full_scenes:
            episode_list = os.listdir(scene)
            full_eps = [os.path.join(scene, x) for x in episode_list]
            for episode in full_eps:
                f = open(episode + '/data.json')
                data = json.load(f)
                frames = np.sort(os.listdir(episode+'/rgb/'))
                frames_full = [episode + '/rgb/' + x for x in frames]
                length = int(np.trunc(len(frames)/frame_interval))
                if 'camera' in data.keys() and len(np.array(data["camera"])) == len(np.array(np.array(frames))):
                    batched_frames = np.array(frames_full)[:length*frame_interval].reshape(length,frame_interval)
                    batched_cps = np.array(data["camera"])[:length*frame_interval].reshape(length,frame_interval)
                    batched_rots = np.array(data["rotations"])[:length*frame_interval].reshape(length,frame_interval)
                    batched_horizon = np.array(data["horizon"])[:length*frame_interval].reshape(length,frame_interval)
                    
                    self.horizon.append(batched_horizon)
                    self.rots.append(batched_rots)
                    self.cps.append(batched_cps)
                    self.episode_frames.append(batched_frames)

        self.horizon = np.vstack(self.horizon)
        self.rots = np.vstack(self.rots)
        self.cps = np.vstack(self.cps)
        self.episode_frames = np.vstack(self.episode_frames)

        # all_objs = []
        # for file_list in file_lists:
        #     if not os.path.exists(file_list):
        #         continue
        #     base_dir = os.path.dirname(file_list)
        #     cat = os.path.basename(base_dir)
        #     with open(file_list, "r") as f:
        #         objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
        #     all_objs.extend(objs)

        # self.all_objs = all_objs
        self.stage = stage

        self.image_to_tensor = get_image_to_tensor_balanced()
        #self.mask_to_tensor = get_mask_to_tensor()
        print(
            "Loading Thor dataset",
            self.base_path,
            "stage",
            stage,
            len(self.full_scenes),
            "objs",
            "type:",
            sub_format,
        )



        self.image_size = image_size
        if sub_format == "dtu":
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        self.sub_format = sub_format
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False

    def __len__(self):
        return len(self.episode_frames)

    def __getitem__(self, index):


        rgb_paths = self.episode_frames[index]
        rot_data = self.rots[index]
        pos_data = self.cps[index]
        horizon_data = self.horizon[index]
        pos_data = [list((x.values())) for x in pos_data]

        all_imgs = []
        all_poses = []
        focal = None
        for idx, (rgb_path, horiz, pos, rot) in enumerate(zip(rgb_paths, horizon_data, pos_data, rot_data)):
            cam_pos = pos#list(np.load(cam_path, allow_pickle=True).item().values())
            img = imageio.imread(rgb_path)

            t = cam_pos
            tilt = horiz#meta_data['cameraHorizon']
            rot_angle = rot['y']#meta_data['rotation']['y']
            r = R.from_euler('yx', [rot_angle, tilt], degrees=True).as_matrix()
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = r#r.transpose()
            pose[:3, 3] = t

            pose = (torch.tensor(pose, dtype=torch.float32))
            #print('pose', pose)
            img_tensor = self.image_to_tensor(img)
            all_imgs.append(img_tensor)
            all_poses.append(pose)
        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        #Need to figure out focal length
        focal = torch.tensor([212.13, 212.13])
        result = {
            #"path": rgb_paths[0],
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
        }
        return result