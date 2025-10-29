import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from transformers import AutoTokenizer

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.async_utils import run
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample

__all__ = ["generate_agent_rollout"]


def _collect_rollout_metrics(data: List[List[Sample]], rollout_id: int, args) -> Dict[str, Any]:
    """收集 rollout 过程中的各种指标

    Args:
        data: list[list[Sample]], 结构是 [group1, group2, ...], 每个 group 包含 n_samples_per_prompt 个 Sample
        rollout_id: rollout 的 ID
        args: 参数

    Returns:
        dict: 包含各种指标的字典
    """
    metrics = {}

    # 统计 stop reason
    stop_reason_count = {"overlong": 0, "overturn": 0, "other": 0, "total": 0}

    # 统计 weight version 分布
    weight_version_count = {}
    total_samples_with_versions = 0

    # 统计 turns
    turns_list = []

    # 遍历所有 samples
    for group in data:
        for sample in group:
            # 统计 stop reason
            stop_reason_count["total"] += 1

            if hasattr(sample, "metadata") and sample.metadata:
                # Stop reason
                stop_reason = sample.metadata.get("stop_reason", "").lower()

                if "overlong" in stop_reason or "over_long" in stop_reason:
                    stop_reason_count["overlong"] += 1
                elif "overturn" in stop_reason or "over_turn" in stop_reason:
                    stop_reason_count["overturn"] += 1
                else:
                    stop_reason_count["other"] += 1

                # Turns
                if "turns" in sample.metadata:
                    turns_list.append(sample.metadata["turns"])
            else:
                stop_reason_count["other"] += 1

            # 统计 weight version
            if hasattr(sample, "weight_versions") and sample.weight_versions:
                total_samples_with_versions += 1
                for version in sample.weight_versions:
                    weight_version_count[version] = weight_version_count.get(version, 0) + 1

    # 计算 stop reason 比例（只记录相对值）
    total = stop_reason_count["total"]
    if total > 0:
        metrics["rollout/stop_reason/overlong_ratio"] = stop_reason_count["overlong"] / total
        metrics["rollout/stop_reason/overturn_ratio"] = stop_reason_count["overturn"] / total
        metrics["rollout/stop_reason/other_ratio"] = stop_reason_count["other"] / total

    # 计算 turns 统计
    if turns_list:
        metrics["rollout/max_turns"] = max(turns_list)
        metrics["rollout/min_turns"] = min(turns_list)
        metrics["rollout/mean_turns"] = sum(turns_list) / len(turns_list)

    # 计算 weight version 分布
    if weight_version_count:
        total_version_occurrences = sum(weight_version_count.values())

        # 1. 纯文本：原始 weight version 分布（绝对比例）
        version_distribution_text = {}
        for version in sorted(
            weight_version_count.keys(), key=lambda x: (int(x) if str(x).isdigit() else 0), reverse=True
        ):
            count = weight_version_count[version]
            ratio = count / total_version_occurrences
            version_distribution_text[version] = ratio

        # 2. 堆积图数据：相对 weight version 占比
        try:
            version_counts_by_int = {}
            for version, count in weight_version_count.items():
                try:
                    version_int = int(version)
                    version_counts_by_int[version_int] = count
                except (ValueError, TypeError):
                    continue

            if version_counts_by_int:
                latest_version = max(version_counts_by_int.keys())
                older_count = 0

                # 最新 9 个 version（使用相对版本号）- 始终记录所有9个,即使是0
                for i in range(9):
                    target_version = latest_version - i
                    count = version_counts_by_int.get(target_version, 0)
                    # 使用相对版本号：latest, latest-1, latest-2, ...
                    relative_label = "latest" if i == 0 else f"latest-{i}"
                    # 总是记录,即使是0
                    metrics[f"rollout/weight_version_stacked/{relative_label}_ratio"] = (
                        count / total_version_occurrences
                    )

                # 更早的版本汇总
                for version_int, count in version_counts_by_int.items():
                    if version_int < latest_version - 8:
                        older_count += count

                # 总是记录 older_ratio,即使是0
                metrics["rollout/weight_version_stacked/older_ratio"] = older_count / total_version_occurrences

                metrics["rollout/weight_version/latest_version"] = latest_version

                # 保存文本供 wandb 上传
                metrics["_version_distribution_text"] = json.dumps(version_distribution_text, indent=2)

        except Exception as e:
            print(f"[WARNING] Failed to generate weight version metrics: {e}")

    # 添加 step 信息
    metrics["rollout/step"] = (
        rollout_id
        if not args.wandb_always_use_train_step
        else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    )

    return metrics


START_ROLLOUT = True


async def batch_get_tokens(
    api_base_url: str, main_keys: List[str], timeout: float = 100.0
) -> Dict[str, Dict[str, Any]]:
    """批量获取 tokens 信息，避免串行调用"""

    async def fetch_single_token(session: aiohttp.ClientSession, main_key: str) -> tuple[str, Dict[str, Any]]:
        url = f"{api_base_url}/tokens"
        payload = {"main_key": main_key}

        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            response.raise_for_status()
            token_data = await response.json()
            return main_key, token_data

    try:
        async with aiohttp.ClientSession() as session:
            # 并行获取所有 tokens
            tasks = [fetch_single_token(session, main_key) for main_key in main_keys]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            token_dict = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"[ERROR] Failed to fetch tokens for {main_keys[i]}: {result}")
                    # 可以选择跳过这个 key 或者重试
                    continue
                else:
                    main_key, token_data = result
                    token_dict[main_key] = token_data

            return token_dict

    except Exception as e:
        print(f"[ERROR] Batch token fetch failed: {e}")
        raise


async def get_rollout_data(
    api_base_url: str,
    num: Optional[int] = None,
    timeout: float = 100.0,
    enable_tito: bool = False,
    sglang_router_ip: str = None,
    sglang_router_port: int = None,
) -> List[Dict[str, Any]]:

    url = f"{api_base_url}/get_rollout_data"
    payload = {}

    if num is not None:
        payload["batch_size"] = num
    print(url)
    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    response.raise_for_status()
                    resp_json = await response.json()
                    if resp_json["success"]:
                        break
                await asyncio.sleep(3)
                if time.time() - start_time > 30:
                    # TODO
                    print("rollout data is not ready, have been waiting for 30 seconds")
                    # Reset start_time to continue waiting or handle timeout differently
                    start_time = time.time()  # Or raise an exception, or return empty list

            data = resp_json["data"]
            meta_info = {}
            if isinstance(data, list):
                if "data" in data[0]:
                    data = [item["data"] for item in data]
            elif isinstance(data, dict):
                if "data" in data:
                    meta_info = data["meta_info"]
                    data = data["data"]
            print(f"Meta info: {meta_info}")
            print("len(data): ", len(data))
            print(data[0].keys())

            required_keys = {"uid", "instance_id", "messages", "reward", "meta_data"}
            rewards_in_agent_rollout = [item["reward"] for item in data]
            print("rewards_in_agent_rollout: ", rewards_in_agent_rollout)
            for item in data:
                if not required_keys.issubset(item.keys()):
                    print("no valid metric")
                    raise ValueError(f"Missing required keys in response item: {item.keys()}")

            # 如果启用了 tito，批量获取所有的 tokens 信息
            if enable_tito and sglang_router_ip and sglang_router_port:
                main_keys = [item["uid"] for item in data]
                tokens_url = f"http://{sglang_router_ip}:{sglang_router_port}"
                print(f"Batch fetching tokens for {len(main_keys)} items")

                try:
                    tokens_dict = await batch_get_tokens(tokens_url, main_keys, timeout)
                    print(f"Successfully fetched tokens for {len(tokens_dict)} items")

                    # 将 tokens 信息添加到相应的数据项中
                    for item in data:
                        uid = item["uid"]
                        if uid in tokens_dict:
                            item["token_info"] = tokens_dict[uid]
                            # if "loss_mask" in item["token_info"]:
                            #     print(f"{len(item["token_info"]["loss_mask"]), sum(item["token_info"]["loss_mask"])}")
                        else:
                            print(f"[WARNING] No token info found for uid: {uid}")

                except Exception as e:
                    print(f"[ERROR] Failed to batch fetch tokens: {e}")
                    # 如果批量获取失败，继续返回原始数据，后续可以回退到串行方式

            return data

    except aiohttp.ClientError as e:
        print(f"[ERROR] Request failed: {e}")
        raise
    except ValueError as ve:
        # print(f"[ERROR] Invalid data format: {ve}")
        raise
    except asyncio.TimeoutError:
        print(f"[ERROR] Request timed out after {timeout} seconds")
        raise


def start_rollout(api_base_url: str, args):

    url = f"{api_base_url}/start_rollout"
    # if args.rollout_input_file is None:
    # raise ValueError("rollout_input_file is required")

    # Convert args to dict and filter out non-serializable objects
    payload = {}
    for key, value in vars(args).items():
        try:
            # Test if the value is JSON segirializable
            import json

            json.dumps(value)
            payload[key] = value
        except (TypeError, ValueError):
            # Skip non-serializable objects
            print(f"[start_rollout] Skipping non-serializable field: {key} (type: {type(value).__name__})")
            continue
    payload["num_process"] = 500

    while True:
        try:
            # resp = requests.post(url, json={"args": payload}, timeout=10)
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            print(f"[start_rollout] Success: {data}")
            return data
        except Exception as e:
            print(f"[start_rollout] Failed to send rollout config: {e}")
            # Add a small delay before retrying to avoid tight loop
            time.sleep(1)


async def generate_agent_rollout(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        engine: the inference engine
        rank: int, the rank of the inference engine
        world_size: int, the world size of the inference engine
        evaluation: bool, whether it and evaluation rollout
    """

    base_url = args.rollout_buffer_url
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    retry_times = 0
    results = []
    status_map = {
        "truncated": Sample.Status.TRUNCATED,
        "completed": Sample.Status.COMPLETED,
        "aborted": Sample.Status.ABORTED,
        "overlong": Sample.Status.TRUNCATED,
        "overturn": Sample.Status.TRUNCATED,
        "format_error": Sample.Status.ABORTED,
    }

    global START_ROLLOUT

    if START_ROLLOUT:
        metadata = data_buffer.get_metadata()
        start_inform = start_rollout(args.rollout_buffer_url, args)
        print(f"start rollout with payload: {start_inform}")
        print(f"start rollout id: {rollout_id}")
        START_ROLLOUT = False

    if args.fetch_trajectory_retry_times == -1:
        print(
            f"⚠️  [get_rollout_data] Fetch trajectory retry times set to -1, will retry indefinitely until sufficient data is collected"
        )

    while args.fetch_trajectory_retry_times == -1 or retry_times < args.fetch_trajectory_retry_times:
        try:
            while (
                len(results)
                < args.min_batch_collection_ratio
                * (args.rollout_batch_size - data_buffer.get_buffer_length())
                * args.n_samples_per_prompt
            ):
                time.sleep(5)
                results.extend(
                    await get_rollout_data(
                        api_base_url=base_url,
                        enable_tito=args.enable_tito,
                        sglang_router_ip=args.sglang_router_ip,
                        sglang_router_port=args.sglang_router_port,
                    )
                )
                print(f"get rollout data with length: {len(results)}")
            break
        except Exception as err:
            retry_times += 1

    print("finally get rollout data with length: ", len(results))

    log_items = {
        "overlong": [],
        "turns": [],
        "overturn": [],
        "abort_times": [], 
        "avg_negative_sample_lengths": [],
        "avg_positive_sample_lengths": [],
        "unnormal_end_by_server_or_format": [],
        "normal_end": [],
        "single_turn_end": [],
    }
    
    sample_results = []
    record_turns = []
    for i in range(0, len(results), args.n_samples_per_prompt):
        records = results[i : i + args.n_samples_per_prompt]
        temp_record = []
        for record in records:
            oai_messages = record["messages"]

            record_turns.append(len([item for item in record["messages"] if item.get("role") == "assistant"]))
            mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
            if args.enable_tito:
                # 检查是否有预获取的 token 信息
                if "token_info" in record:
                    # 使用预获取的 token 信息
                    token_response = record["token_info"]
                    print(f"Using pre-fetched token info for uid: {record['uid']}")
                else:
                    # 回退到串行调用（如果批量获取失败）
                    print(f"[FALLBACK] Fetching token info individually for uid: {record['uid']}")
                    try:
                        token_response = requests.post(
                            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/tokens",
                            json={"main_key": record["uid"]},
                        ).json()
                    except Exception as e:
                        print(f"[FALLBACK] Failed to fetch token info individually for uid: {record['uid']}: {e}")
                        token_response = None

                # print("decode result", tokenizer.decode(token_response["input_ids"]))
                # print("message apply", tokenizer.apply_chat_template([record["messages"]], tokenize=False))
                if "input_ids" in token_response:
                    token_ids = token_response["input_ids"]
                    loss_mask = token_response["loss_mask"]
                    rollout_logprobs = token_response["logprobs"]
                    response_length = mask_generator.get_response_lengths([loss_mask])[0]
                    # with open("debug_token_ids.jsonl", "a") as f:
                    #     data_temp = {"token_ids": token_ids, "messages": record["messages"], "uid": record["uid"],"loss_mask": loss_mask}
                    #     f.write(f"{json.dumps(data_temp)}\n")

                else:
                    # oai_messages = tokenizer.apply_chat_template([record["messages"]], tokenize=False)
                    # print(mask_generator.get_text_from_loss_mask(token_ids, loss_mask))
                    token_ids, loss_mask = mask_generator.get_loss_mask(record["messages"])
                    response_length = mask_generator.get_response_lengths([loss_mask])[0]
                    rollout_logprobs = [0] * response_length
                    loss_mask = [0] * response_length
            else:
                # oai_messages = tokenizer.apply_chat_template([record["messages"]], tokenize=False)
                # print(mask_generator.get_text_from_loss_mask(token_ids, loss_mask))
                token_ids, loss_mask = mask_generator.get_loss_mask(record["messages"])
                response_length = mask_generator.get_response_lengths([loss_mask])[0]
            # print("loss mask", sum(loss_mask), len(loss_mask), max(loss_mask))
            if sum(loss_mask) == 0:
                response_length = len(loss_mask)
            loss_mask = loss_mask[-response_length:]
            # import json
            # with open("debug_agent_rollout_loss_mask.jsonl", "a") as f:
            #     data_temp = {
            #         "token_ids": token_ids,
            #         "messages": record["messages"],
            #         "uid": record["uid"],
            #         "loss_mask": loss_mask,
            #         "response_length": response_length,
            #     }
            #     f.write(json.dumps(data_temp) + "\n")

            temp_record.append(
                Sample(
                    index=record["instance_id"],
                    prompt=record["uid"],
                    tokens=token_ids,
                    response_length=response_length,
                    rollout_log_probs=rollout_logprobs[-response_length:] if args.enable_tito else None,
                    reward=record["reward"],
                    status=status_map[record["status"]] if "status" in record else Sample.Status.COMPLETED,
                    loss_mask=loss_mask,
                    weight_versions=record["meta_data"].get("weight_versions", []),
                    metadata={**record["meta_data"]},
                )
            )

            log_items["turns"].append(len(oai_messages) // 2)
            log_items["overturn"].append(record.get("overturn", False))
            log_items["overlong"].append(record.get("overlong", False))
            log_items["abort_times"].append(record.get("abort_times", 0))

            log_items["unnormal_end_by_server_or_format"].append(record.get("unnormal_end_by_server_or_format", False))
            log_items["normal_end"].append(record.get("normal_end", False))
            log_items["single_turn_end"].append(record.get("single_turn_end", False))

            if record["raw_reward"] <= 0:
                log_items["avg_negative_sample_lengths"].append(len(token_ids))
            else:
                log_items["avg_positive_sample_lengths"].append(len(token_ids))

        if args.truncate_negative_samples:
            group_results = temp_record
            positive_items = [item for item in group_results if item.reward > 0]
            negative_items = [item for item in group_results if item.reward <= 0]
            if len(positive_items) > 0 and len(negative_items) > 0:
                _group_results = []
                max_pos_len = max([len(sample.tokens) for sample in positive_items])
                max_pos_len_threshold = int(max_pos_len * 1.4)
                for sample in negative_items:
                    if len(sample.tokens) > max_pos_len_threshold:
                        exceeded_len = len(sample.tokens) - max_pos_len_threshold
                        tokens = sample.tokens[:-exceeded_len]
                        loss_mask = sample.loss_mask[:-exceeded_len]
                        new_sample = Sample(
                            index=sample.index,
                            prompt=sample.prompt,
                            tokens=tokens,
                            response_length=len(loss_mask),
                            loss_mask=loss_mask,
                            reward=sample.reward,
                            status=sample.status,
                            metadata=sample.metadata,
                            rollout_log_probs=sample.rollout_log_probs,
                        )
                        _group_results.append(new_sample)
                    else:
                        _group_results.append(sample)
                _group_results.extend(positive_items)
                group_results = _group_results
        

        sample_results.append(temp_record)

    data_buffer.add_samples(sample_results)
    final_return_results = data_buffer.get_samples(args.rollout_batch_size)

    print("len(final_return_results): ", len(final_return_results))

    # 收集 metrics
    if args.use_wandb:
        metrics = _collect_rollout_metrics(final_return_results, rollout_id, args)
        import json
        import os

        import wandb

        # 上传原始 weight version 分布文本(仅用于调试查看)
        if "_version_distribution_text" in metrics:
            # 打印到日志
            print(f"\n=== Rollout {rollout_id} Weight Version Distribution ===")
            print(metrics["_version_distribution_text"])
            print("=" * 60)

            # 保存到文件并上传到 wandb (与 run_id 绑定)
            run_id = wandb.run.id if wandb.run else "unknown"
            metrics_dir = os.path.join(wandb.run.dir, "rollout_metrics") if wandb.run else "rollout_metrics"
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_log_file = os.path.join(metrics_dir, f"metrics_{run_id}.jsonl")
            
            metric_entry = {
                "rollout_id": rollout_id,
                "step": metrics.get("rollout/step", rollout_id),
                "weight_version_distribution": metrics["_version_distribution_text"],
                "stop_reason": {
                    "overlong_ratio": metrics.get("rollout/stop_reason/overlong_ratio", 0),
                    "overturn_ratio": metrics.get("rollout/stop_reason/overturn_ratio", 0),
                    "other_ratio": metrics.get("rollout/stop_reason/other_ratio", 0),
                },
                "turns": {
                    "max": metrics.get("rollout/max_turns", 0),
                    "min": metrics.get("rollout/min_turns", 0),
                    "mean": metrics.get("rollout/mean_turns", 0),
                },
            }

            with open(metrics_log_file, "a") as f:
                f.write(json.dumps(metric_entry) + "\n")

            wandb.save(metrics_log_file)

            del metrics["_version_distribution_text"]  # 删除临时字段

        wandb.log(metrics)

        # ---- temp
        num_items = max([len(x) for x in log_items.values()])
        num_items = num_items if num_items > 0 else 1
        log_dict = {
            "rollout/overlong": sum(log_items["overlong"]) / num_items,
            "rollout/max_turns": max(log_items["turns"]),
            "rollout/min_turns": min(log_items["turns"]),
            "rollout/avg_turns": sum(log_items["turns"]) / num_items,
            "rollout/overturn": sum(log_items["overturn"]) / num_items,
            "rollout/abort_ratio": sum([x > 0 for x in log_items["abort_times"]]) / num_items,
            "rollout/abort_ratio2": sum([x > 1 for x in log_items["abort_times"]]) / num_items,

            "rollout/avg_negative_sample_lengths": sum(log_items["avg_negative_sample_lengths"]) / max(1, len(log_items["avg_negative_sample_lengths"])),
            "rollout/avg_positive_sample_lengths": sum(log_items["avg_positive_sample_lengths"]) / max(1, len(log_items["avg_positive_sample_lengths"])),
            
            "rollout/unnormal_end_by_server_or_format": sum(log_items["unnormal_end_by_server_or_format"]) / num_items,
            "rollout/normal_end": sum(log_items["normal_end"]) / num_items,
            "rollout/single_turn_end": sum(log_items["single_turn_end"]) / num_items
        }
        wandb.log(log_dict)

    # 返回空 metrics，因为已经在上面 log 过了，避免重复
    return RolloutFnTrainOutput(samples=final_return_results, metrics={})


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise NotImplementedError("Evaluation is not implemented for agent rollout")
    return run(generate_agent_rollout(args, rollout_id, data_buffer))
