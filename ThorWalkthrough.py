import os
from tqdm import tqdm
import random
from ai2thor.controller import Controller
from PIL import Image
import numpy as np
from ai2thor.platform import CloudRendering
import cv2
import json
from multiprocessing import Pool
import prior
import copy
import time
import argparse
import GPUtil
#from utils.objectnav import target_object_types

target_object_types = [
    "AlarmClock",
    "Apple",
    "BaseballBat",
    "BasketBall",
    "Bowl",
    "GarbageCan",
    "HousePlant",
    "Laptop",
    "Mug",
    "RemoteControl",
    "SprayBottle",
    "Television",
    "Vase"
]


NUM_STEPS = 5e7
SIZE = 256
ROTATION_STEP = 30
ANGLES = list(range(0, 360, ROTATION_STEP))
DATASET = prior.load_dataset("procthor-10k")
SCENES = DATASET["train"]
PATH = "rollouts2/ProcTHORObjectNavRGB50M"
NUM_EP_PER_SCENE = 50
NUM_SCENES = 500

random.seed("Optimus RGB ProcTHOR Guided ObjectNav Data Collection 150M")


def collect_scene_episodes(scene_id):
    start_time = time.time()
    scene = SCENES[scene_id]
    print(scene_id % args.gpu)
    env = Controller(
        agentMode="default",
        commit_id="6c0aafb84ca6ffe902768411a7b7b0c895de40be",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene,
        gridSize=0.25,
        snapToGrid=False,
        movementGaussianSigma=0.005,
        rotateStepDegrees=ROTATION_STEP,
        rotateGaussianSigma=0.5,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width=256,
        height=256,
        fieldOfView=60,
        platform=CloudRendering,
        gpu_device = scene_id % args.gpu,
        #x_display=":0.{}".format(scene_id % args.gpu)
    )
    for ep in range(NUM_EP_PER_SCENE):
        try:
            collect_episode(env, scene, ep, scene_id)
        except Exception as e:
            print("RESETTING THE SCENE,", scene, "because of", str(e))
            env = Controller(
                    agentMode="default",
                    commit_id="6c0aafb84ca6ffe902768411a7b7b0c895de40be",
                    makeAgentsVisible=False,
                    visibilityDistance=1.5,
                    scene=scene,
                    gridSize=0.25,
                    snapToGrid=False,
                    movementGaussianSigma=0.005,
                    rotateStepDegrees=ROTATION_STEP,
                    rotateGaussianSigma=0.5,
                    renderDepthImage=True,
                    renderInstanceSegmentation=True,
                    width=256,
                    height=256,
                    fieldOfView=60,
                    platform=CloudRendering,
                    gpu_device = scene_id % args.gpu,
                    #x_display=":0.{}".format(scene_id % args.gpu)
    )
            env.reset(scene)

    env.stop()

    end_time = time.time()
    msg = "Scene {} took {:.2f} minutes".format(scene_id, (end_time - start_time)/60)
    return msg


def collect_episode(env, scene, ep, scene_id):
    episode_dir = os.path.join(PATH, "House_{:06d}".format(scene_id))
    episode_path = os.path.join(episode_dir, "Episode_{:06d}".format(ep))
    if os.path.exists(os.path.join(episode_path, "rgb")):
        print('already made episode')
        return
    # reset scene and get reachable positions
    env.reset(scene=scene)
    reachable_positions = env.step(action="GetReachablePositions").metadata["actionReturn"]
    # choose starting position
    root_pos = random.choice(reachable_positions)
    root_angle = random.choice(ANGLES)

    env.step(action="Teleport", position=root_pos, rotation=dict(x=0, y=root_angle, z=0), horizon=0)
    # take an action to get correct lighting of first frame
    env.step(action="RotateRight")

    frames = [np.uint8(env.last_event.frame)]
    positions = [env.last_event.metadata["agent"]["position"]]
    rotations = [env.last_event.metadata["agent"]["rotation"]]
    cps = [env.last_event.metadata["agent"]["cameraPosition"]]
    horizon = [env.last_event.metadata["agent"]['cameraHorizon']]
    visible = [get_visible(env)]
    #in_range = [get_in_range(env)]
    actions = []

    # get all the objectnav targets in the scene
    all_scene_targets = [x for x in env.last_event.metadata['objects'] if x['objectType'] in target_object_types]
    shuffled_targets = shuffle_targets(all_scene_targets)
    instances = pick_instances(shuffled_targets)
    seen_scene_targets = []

    while len(instances) > 0:
        target = instances[0]
        a = env.step(action="ObjectNavExpertAction", objectId=target['objectId']).metadata["actionReturn"]

        if a is None:
            instances.pop(0)
            continue
        # frames.append(np.uint8(env.last_event.frame))
        # positions.append(env.last_event.metadata["agent"]["position"])
        # rotations.append(env.last_event.metadata["agent"]["rotation"])
        # visible.append(get_visible(env))
        # in_range.append(get_in_range(env))
        while a is not None:
            # Roll out env
            GPUtil.showUtilization()
            actions.append(a)
            env.step(action=a)
            frames.append(np.uint8(env.last_event.frame))
            positions.append(env.last_event.metadata["agent"]["position"])
            rotations.append(env.last_event.metadata["agent"]["rotation"])
            cps.append(env.last_event.metadata["agent"]["cameraPosition"])
            horizon.append(env.last_event.metadata["agent"]["cameraHorizon"])
            visible.append(get_visible(env))
            #in_range.append(get_in_range(env))
            a = env.step(action="ObjectNavExpertAction", objectId=target['objectId']).metadata["actionReturn"]

            # If the episode is longer than 2000 steps, kill it
            if len(frames) > 300:
                break

        # If the agent reaches the target looking up or down correct its gaze to look straight
        while not -0.5 < env.last_event.metadata["agent"]["cameraHorizon"] < 0.5:
            if env.last_event.metadata["agent"]["cameraHorizon"] < 0.0:
                a = "LookDown"
            else:
                a = "LookUp"
            actions.append(a)
            env.step(action=a)
            frames.append(np.uint8(env.last_event.frame))
            positions.append(env.last_event.metadata["agent"]["position"])
            rotations.append(env.last_event.metadata["agent"]["rotation"])
            cps.append(env.last_event.metadata["agent"]["cameraPosition"])
            horizon.append(env.last_event.metadata["agent"]["cameraHorizon"])
            visible.append(get_visible(env))
            #in_range.append(get_in_range(env))

        seen_scene_targets.append(instances.pop(0))

    # if for some reason no targets were found do not save episode
    if len(seen_scene_targets) == 0:
        return

    actions.append("Stop")
    # make_video(frames, "video")

    # if the episode is shorter than 100 frames do not save it
    if len(frames) < 100:
        return

    episode_dir = os.path.join(PATH, "House_{:06d}".format(scene_id))
    episode_path = os.path.join(episode_dir, "Episode_{:06d}".format(ep))
    os.makedirs(os.path.join(episode_path, "rgb"), exist_ok=True)

    image_paths = []
    for frame_idx, frame in enumerate(frames):
        im = Image.fromarray(frame)
        image_path = os.path.join(episode_path, "rgb/Frame_{:06d}.png".format(frame_idx))
        im.save(image_path)
        image_paths.append("rgb/Frame_{:06d}.png".format(frame_idx))

    data = {"positions": positions, "rotations": rotations, "camera": cps, "horizon": horizon, "actions": actions,
            "visible": visible, "target": seen_scene_targets[-1]['objectType'],
            "rgb_frames": image_paths}
    with open(os.path.join(episode_path, "data.json"), "w") as f:
        json.dump(data, f)


def get_object_data(env):
    objs = env.last_event.metadata["objects"]
    obj_ids = set([o['objectId'] for o in objs])
    box_ids = set(env.last_event.instance_detections2D.keys())
    for o in objs:
        if o['objectId'] in box_ids:
            o["bbox"] = tuple([int(c) for c in env.last_event.instance_detections2D[o['objectId']]])
    for b in box_ids - obj_ids:
        objs.append({"bbox": tuple([int(c) for c in env.last_event.instance_detections2D[b]]),
                     "objectId": b, "name": b.split(" ")[0]})
    return objs


def get_in_range(env):
    in_range = set()
    objects = env.last_event.metadata['objects']
    for o in objects:
        if o['objectType'] in target_object_types and o['visible']:
            in_range.add(o['objectType'])
    return list(in_range)


def get_visible(env):
    return dict(env.last_event.instance_detections2D)


def shuffle_targets(targets):
    target_copy = copy.deepcopy(targets)
    random.shuffle(target_copy)
    while any(
            [np.sqrt((a["position"]["x"] - b["position"]["x"])**2 + (a["position"]["z"] - b["position"]["z"])**2) < 1.0
             for a, b in zip(target_copy[:-1], target_copy[1:])]):
        random.shuffle(target_copy)
    return target_copy


def pick_instances(targets):
    instances = []
    for t in targets:
        if t["objectType"] not in [i["objectType"] for i in instances]:
            instances.append(t)
    return instances


def make_video(frames, out_file):

    # initialize video writer
    out = cv2.VideoWriter("%s.mp4" % out_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (256, 256), isColor=True)

    # new frame after each addition of water
    for frame in frames:
        out.write(frame)

    # close out the video writer
    out.release()


def collect_data_main(args):
    scenes = list(range(NUM_SCENES))#list(range(len(SCENES)))
    with Pool(args.gpu*2) as p:
        for msg in tqdm(p.imap_unordered(collect_scene_episodes, scenes), total=len(scenes)):
            print(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    collect_data_main(args)