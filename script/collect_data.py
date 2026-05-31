import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file, config_name="config.yml"):
    robot_config_file = os.path.join(robot_file, config_name)
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    embodiment_config_name = args.get("embodiment_config", "config.yml")
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"], embodiment_config_name)
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"], embodiment_config_name)

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        # Optional episode-range mode so several workers can collect disjoint
        # ranges in parallel (on different GPUs). Set via env vars:
        #   COLLECT_EP_START / COLLECT_EP_END  -> [start, end) episode range
        #   COLLECT_SEED_OFFSET                -> per-worker offset for the fresh
        #                                         re-plan seed pool (keeps replanned
        #                                         episodes unique across workers)
        ep_start = int(os.environ.get("COLLECT_EP_START", 0))
        ep_end = int(os.environ.get("COLLECT_EP_END", args["episode_num"]))
        seed_offset = int(os.environ.get("COLLECT_SEED_OFFSET", 0))
        range_mode = ("COLLECT_EP_START" in os.environ) or ("COLLECT_EP_END" in os.environ)

        # In range mode each worker keeps its own scene_info shard (merged later)
        # to avoid racing on the shared json file.
        if range_mode:
            info_file_path = os.path.join(args["save_path"], f"scene_info_part_{ep_start}_{ep_end}.json")
        else:
            info_file_path = os.path.join(args["save_path"], "scene_info.json")
        if not os.path.exists(info_file_path):
            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump({}, file, ensure_ascii=False)

        seed_file_path = os.path.join(args["save_path"], "seed.txt")

        # Some tasks (e.g. contact-rich mid-air hand-offs) do not always
        # reproduce their seed-phase result when the recorded trajectory is
        # replayed open-loop, so a single failed replay must NOT abort the whole
        # collection. We retry the recorded trajectory a few times and, if it
        # still cannot be reproduced, fall back to collecting a fresh online
        # plan with a new seed (same procedure the seed phase already validated).
        REPLAY_RETRY = 2
        MAX_REPLAN = 40
        next_fresh_seed = (max(seed_list) + 1 if len(seed_list) > 0 else 0) + seed_offset
        used_seeds = set(seed_list)

        def collect_once(ep_idx, seed, need_plan):
            """Run one full collection attempt; returns (success, info)."""
            args["need_plan"] = need_plan
            TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=seed, **args)
            if need_plan:
                args["left_joint_path"] = []
                args["right_joint_path"] = []
            else:
                traj_data = TASK_ENV.load_tran_data(ep_idx)
                args["left_joint_path"] = traj_data["left_joint_path"]
                args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)
            info = TASK_ENV.play_once()
            TASK_ENV.close_env(clear_cache=((ep_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()
            success = TASK_ENV.plan_success and TASK_ENV.check_success()
            return success, info

        for episode_idx in range(ep_start, ep_end):
            if exist_hdf5(episode_idx):
                continue
            print(f"\033[34mTask name: {args['task_name']} | collecting episode {episode_idx}\033[0m")

            collected_info = None

            # 1) Replay the pre-planned trajectory for this index (with retries).
            for attempt in range(REPLAY_RETRY):
                try:
                    ok, info = collect_once(episode_idx, seed_list[episode_idx], need_plan=False)
                except Exception as e:
                    print(f"[Collect] episode {episode_idx} replay attempt {attempt + 1} raised: {e}")
                    ok = False
                if ok:
                    collected_info = info
                    break
                print(f"[Collect] episode {episode_idx} replay did not reproduce check_success "
                      f"(attempt {attempt + 1}/{REPLAY_RETRY}, seed={seed_list[episode_idx]})")

            # 2) Fall back to online re-planning with fresh seeds until one works.
            replan_tries = 0
            while collected_info is None and replan_tries < MAX_REPLAN:
                while next_fresh_seed in used_seeds:
                    next_fresh_seed += 1
                seed = next_fresh_seed
                used_seeds.add(seed)
                next_fresh_seed += 1
                replan_tries += 1
                try:
                    ok, info = collect_once(episode_idx, seed, need_plan=True)
                except Exception as e:
                    print(f"[Collect] episode {episode_idx} replan seed={seed} raised: {e}")
                    ok = False
                if ok:
                    print(f"[Collect] episode {episode_idx} re-planned successfully with new seed={seed}")
                    seed_list[episode_idx] = seed
                    TASK_ENV.save_traj_data(episode_idx)
                    # Skip rewriting the shared seed.txt in range mode (parallel
                    # workers would race on it); the per-episode _traj_data is
                    # already persisted above.
                    if not range_mode:
                        with open(seed_file_path, "w") as file:
                            for sed in seed_list:
                                file.write("%s " % sed)
                    collected_info = info
                else:
                    print(f"[Collect] episode {episode_idx} replan seed={seed} failed "
                          f"({replan_tries}/{MAX_REPLAN})")

            if collected_info is None:
                raise RuntimeError(
                    f"Failed to collect episode {episode_idx} after {REPLAY_RETRY} replay "
                    f"retries and {MAX_REPLAN} replans")

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)
            info_db[f"episode_{episode_idx}"] = collected_info
            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

        # In range mode, language-instruction generation is run once externally
        # after all workers finish and the scene_info shards are merged.
        if not range_mode:
            command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
            os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
