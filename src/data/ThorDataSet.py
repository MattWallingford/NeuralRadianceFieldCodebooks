import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2
from util import get_image_to_tensor_balanced, get_mask_to_tensor
from scipy.spatial.transform import Rotation as R

class ThorDataset(torch.utils.data.Dataset):
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
            scenes = scenes[8*int(len(scenes)/10):9*int(len(scenes)/10)]
        if stage == "test":
            #print(9*int(len(scenes)/10))
            scenes = os.listdir('../kaolin/examples/tutorial/thor_random_auto_val2/')#scenes[9*int(len(scenes)/10):]
        
        # if stage == "train":
        #     scenes = list(np.arange(1,26)) + list(np.arange(201,226)) + list(np.arange(301,326)) #+ list(np.arange(401,426))
        # elif stage == "val":
        #     scenes = list(np.arange(26,29)) + list(np.arange(226,229)) + list(np.arange(326,329))
        # elif stage == "test":
        #     scenes = list(np.arange(26,29)) + list(np.arange(226,229)) + list(np.arange(326,329)) #+ list(np.arange(426,429))

        # for scene in scenes:
        #     self.full_scenes.append(os.path.join(self.base_path, 'FloorPlan' + str(scene)))
        self.full_scenes = [os.path.join(self.base_path, x) for x in scenes]
        
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
        return len(self.full_scenes)

    def __getitem__(self, index):
        #cat, root_dir = self.all_objs[index]
        scene_path = self.full_scenes[index]
        scene_list = os.listdir(self.full_scenes[index])
        metadata_paths = [os.path.join(scene_path, x) for x in scene_list if 'meta_data' in x]
        metadata_paths = sorted(metadata_paths)



        #print(camera_paths)
        #change this for older Thor datasets
        if True:
            rgb_paths = [x[:-13]+'.jpg' for x in metadata_paths]
            cam_paths = [x[:-13] +'cp.npy' for x in metadata_paths]
        else:
            rgb_paths = [x[:-7]+'.jpg' for x in metadata_paths]
        #print(rgb_paths[0])
        all_imgs = []
        all_poses = []
        focal = None
        for idx, (rgb_path, metadata_path, cam_path) in enumerate(zip(rgb_paths, metadata_paths, cam_paths)):
            meta_data = np.load(metadata_path, allow_pickle=True).item()
            cam_pos = list(np.load(cam_path, allow_pickle=True).item().values())
            img = imageio.imread(rgb_path)
            t = cam_pos
            tilt = meta_data['cameraHorizon']
            rot_angle = meta_data['rotation']['y']
            r = R.from_euler('yx', [-rot_angle, -tilt], degrees=True).as_matrix()
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = r.transpose()
            pose[:3, 3] = t

            pose = (torch.tensor(pose, dtype=torch.float32))

            img_tensor = self.image_to_tensor(img)

            all_imgs.append(img_tensor)
            all_poses.append(pose)
        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        #Need to figure out focal length
        focal = torch.tensor([212.13, 212.13])
        result = {
            "path": scene_path,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
        }
        return result