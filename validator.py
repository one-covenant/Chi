"""
what_measuring: TPS achieved by miners' submitted train.py on a public benchmark.
miner_endpoints: Miners commit a Docker image URL to chain (optional fingerprint in JSON).
request_format: Validator calls Actor.evaluate(task_id, seed, model_url, data_url, steps, batch_size, timeout).
response_format: {"tps": float, "total_tokens": int, "wall_time_seconds": float, "success": bool, "error": str | None}
scoring_criteria: Median TPS across N runs; failures score 0; best score per coldkey; weights normalized.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import bittensor as bt
import click
from bittensor_wallet import Wallet

try:
    from affinetes import env as af_env
except Exception:
    af_env = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

HEARTBEAT_TIMEOUT = 600  # seconds
BLOCK_TIME_SECONDS = 12

# Evaluation config (public + deterministic)
EVAL_RUNS = int(os.getenv("EVAL_RUNS", "3"))
EVAL_TIMEOUT_SECONDS = int(os.getenv("EVAL_TIMEOUT_SECONDS", "600"))
EVAL_STEPS = int(os.getenv("EVAL_STEPS", "5"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "8"))
EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "4"))

# Public benchmark artifacts (no secret eval sets)
BENCHMARK_MODEL_URL = os.getenv("BENCHMARK_MODEL_URL", "").strip()
BENCHMARK_DATA_URL = os.getenv("BENCHMARK_DATA_URL", "").strip()

# Basilica container limits
BASILICA_CPU_LIMIT = os.getenv("BASILICA_CPU_LIMIT", "2000m")
BASILICA_MEM_LIMIT = os.getenv("BASILICA_MEM_LIMIT", "8Gi")

# TODO: Pin benchmark URLs to immutable content hashes for stronger reproducibility.
# TODO: Add optional Hippius provenance checks for benchmark artifacts.


@dataclass
class EvalResult:
    uid: int
    hotkey: str
    coldkey: str
    image_url: str
    fingerprint: str | None
    tps_scores: list[float]
    success_runs: int
    error: str | None = None

    @property
    def median_tps(self) -> float:
        if not self.tps_scores:
            return 0.0
        sorted_scores = sorted(self.tps_scores)
        return sorted_scores[len(sorted_scores) // 2]


def heartbeat_monitor(last_heartbeat: list[float], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        time.sleep(5)
        if time.time() - last_heartbeat[0] > HEARTBEAT_TIMEOUT:
            logger.error(
                "No heartbeat detected in the last 600 seconds. Restarting process.")
            logging.shutdown()
            os.execv(sys.executable, [sys.executable] + sys.argv)


def parse_commitment(commitment: str) -> tuple[str | None, str | None]:
    if not commitment:
        return None, None
    commitment = commitment.strip()
    if commitment.startswith("{"):
        try:
            data = json.loads(commitment)
            image = data.get("image") or data.get("image_url")
            fingerprint = data.get("fingerprint") or data.get("fp")
            image_value = image.strip() if image else None
            fingerprint_value = fingerprint.strip() if isinstance(fingerprint, str) else None
            return image_value, fingerprint_value
        except Exception:
            return None, None
    if commitment.startswith("image="):
        return commitment.split("=", 1)[1].strip() or None, None
    if commitment.startswith("docker://"):
        return commitment.replace("docker://", "", 1).strip() or None, None
    return commitment, None


def get_weights_rate_limit(subtensor: bt.Subtensor, netuid: int) -> int:
    try:
        hparams = subtensor.get_subnet_hyperparameters(netuid)
        rate_limit = getattr(hparams, "weights_rate_limit", None)
        if rate_limit is not None:
            return int(rate_limit)
    except Exception as exc:
        logger.warning(f"Failed to read weights_rate_limit: {exc}")
    return 100


def make_seed(block_number: int, uid: int, run_idx: int) -> str:
    return f"{block_number}:{uid}:{run_idx}"


async def evaluate_miner(
    uid: int,
    hotkey: str,
    coldkey: str,
    image_url: str,
    fingerprint: str | None,
    current_block: int,
    semaphore: asyncio.Semaphore,
) -> EvalResult:
    if af_env is None:
        return EvalResult(
            uid=uid,
            hotkey=hotkey,
            coldkey=coldkey,
            image_url=image_url,
            fingerprint=fingerprint,
            tps_scores=[],
            success_runs=0,
            error="affinetes not installed",
        )

    env_vars = {}
    # if os.getenv("CHUTES_API_KEY"):
    #     env_vars["CHUTES_API_KEY"] = os.getenv("CHUTES_API_KEY")

    try:
        env = af_env.load_env(
            mode="basilica",
            image=image_url,
            cpu_limit=BASILICA_CPU_LIMIT,
            mem_limit=BASILICA_MEM_LIMIT,
            env_vars=env_vars,
        )
    except Exception as exc:
        return EvalResult(
            uid=uid,
            hotkey=hotkey,
            coldkey=coldkey,
            image_url=image_url,
            fingerprint=fingerprint,
            tps_scores=[],
            success_runs=0,
            error=f"load_env failed: {exc}",
        )

    tps_scores: list[float] = []
    success_runs = 0
    error_message = None

    try:
        for run_idx in range(EVAL_RUNS):
            seed = make_seed(current_block, uid, run_idx)
            payload = dict(
                task_id=run_idx,
                seed=seed,
                model_url=BENCHMARK_MODEL_URL,
                data_url=BENCHMARK_DATA_URL,
                steps=EVAL_STEPS,
                batch_size=EVAL_BATCH_SIZE,
                timeout=EVAL_TIMEOUT_SECONDS,
            )

            try:
                async with semaphore:
                    result = await asyncio.wait_for(
                        env.evaluate(**payload),
                        timeout=EVAL_TIMEOUT_SECONDS + 30,
                    )
            except asyncio.TimeoutError:
                error_message = "timeout"
                tps_scores.append(0.0)
                continue
            except Exception as exc:
                error_message = f"evaluate failed: {exc}"
                tps_scores.append(0.0)
                continue

            parsed = parse_eval_result(result)
            if parsed["success"]:
                success_runs += 1
                tps_scores.append(parsed["tps"])
            else:
                error_message = parsed.get("error") or "miner returned failure"
                diagnostics = parsed.get("diagnostics")
                if diagnostics:
                    logger.info(
                        "uid=%s run=%s diagnostics=%s",
                        uid,
                        run_idx,
                        diagnostics,
                    )
                tps_scores.append(0.0)
    finally:
        try:
            await env.cleanup()
        except Exception as exc:
            logger.warning(f"Cleanup failed for uid={uid}: {exc}")

    return EvalResult(
        uid=uid,
        hotkey=hotkey,
        coldkey=coldkey,
        image_url=image_url,
        fingerprint=fingerprint,
        tps_scores=tps_scores,
        success_runs=success_runs,
        error=error_message,
    )


def parse_eval_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        tps = float(result.get("tps", 0.0) or 0.0)
        success = bool(result.get("success", tps > 0))
        return {
            "tps": tps,
            "success": success,
            "error": result.get("error"),
            "diagnostics": result.get("diagnostics"),
        }
    return {"tps": 0.0, "success": False, "error": "invalid result", "diagnostics": None}


def normalize_weights(scores: dict[int, float]) -> dict[int, float]:
    total = sum(scores.values())
    if total <= 0:
        return {uid: 0.0 for uid in scores}
    return {uid: score / total for uid, score in scores.items()}


async def evaluate_and_score(
    subtensor: bt.Subtensor,
    metagraph: bt.Metagraph,
    netuid: int,
    current_block: int,
) -> dict[int, float]:
    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)

    miners: list[tuple[int, str, str, str, str | None]] = []
    fingerprint_map: dict[str, list[tuple[int, str]]] = {}
    for uid, hotkey in enumerate(metagraph.hotkeys):
        try:
            commitment = subtensor.get_commitment(netuid=netuid, uid=uid)
        except Exception as exc:
            logger.debug(f"Commitment read failed for uid={uid}: {exc}")
            continue

        image_url, fingerprint = parse_commitment(
            commitment) if commitment else (None, None)
        if not image_url:
            continue

        coldkey = metagraph.coldkeys[uid]
        miners.append((uid, hotkey, coldkey, image_url, fingerprint))
        if fingerprint:
            fingerprint_map.setdefault(fingerprint, []).append((uid, hotkey))

    if not miners:
        logger.warning("No miner image commitments found; skipping scoring.")
        return {}

    logger.info(f"Evaluating {len(miners)} miners via Basilica.")

    tasks = [
        evaluate_miner(
            uid,
            hotkey,
            coldkey,
            image_url,
            fingerprint,
            current_block,
            semaphore,
        )
        for uid, hotkey, coldkey, image_url, fingerprint in miners
    ]
    results = await asyncio.gather(*tasks)

    scores_by_uid: dict[int, float] = {}
    best_by_coldkey: dict[str, tuple[int, float]] = {}

    for result in results:
        score = result.median_tps
        scores_by_uid[result.uid] = score

        best = best_by_coldkey.get(result.coldkey)
        if best is None or score > best[1]:
            best_by_coldkey[result.coldkey] = (result.uid, score)

        logger.info(
            "uid=%s hotkey=%s tps=%s runs=%s fp=%s error=%s",
            result.uid,
            result.hotkey,
            f"{score:.2f}",
            result.success_runs,
            result.fingerprint[:12] if result.fingerprint else "none",
            result.error,
        )

    for fingerprint, owners in fingerprint_map.items():
        if len(owners) > 1:
            owners_str = ", ".join(
                [f"{uid}:{hotkey[:8]}" for uid, hotkey in owners])
            logger.warning(
                "Potential copy detected (same fingerprint): fp=%s owners=%s",
                fingerprint[:16],
                owners_str,
            )

    # Keep best score per coldkey; others set to 0
    filtered_scores: dict[int, float] = {uid: 0.0 for uid in scores_by_uid}
    for coldkey, (uid, score) in best_by_coldkey.items():
        filtered_scores[uid] = score

    # TODO: Consider stronger anti-gaming mechanisms if weight-copying emerges.
    return normalize_weights(filtered_scores)


@click.command()
@click.option(
    "--network",
    default=lambda: os.getenv("NETWORK", "finney"),
    help="Network to connect to (finney, test, local)",
)
@click.option(
    "--netuid",
    type=int,
    default=lambda: int(os.getenv("NETUID", "1")),
    help="Subnet netuid",
)
@click.option(
    "--coldkey",
    default=lambda: os.getenv("WALLET_NAME", "default"),
    help="Wallet name",
)
@click.option(
    "--hotkey",
    default=lambda: os.getenv("HOTKEY_NAME", "default"),
    help="Hotkey name",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"],
                      case_sensitive=False),
    default=lambda: os.getenv("LOG_LEVEL", "INFO"),
    help="Logging level",
)
def main(network: str, netuid: int, coldkey: str, hotkey: str, log_level: str) -> None:
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    logger.info(
        f"Starting TPS validator on network={network}, netuid={netuid}")

    if not BENCHMARK_MODEL_URL or not BENCHMARK_DATA_URL:
        logger.warning(
            "BENCHMARK_MODEL_URL or BENCHMARK_DATA_URL not set; "
            "miners must handle empty URLs or validator should be configured."
        )

    last_heartbeat = [time.time()]
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat_monitor, args=(
            last_heartbeat, stop_event), daemon=True
    )
    heartbeat_thread.start()

    wallet = Wallet(name=coldkey, hotkey=hotkey)
    subtensor = bt.Subtensor(network=network)
    metagraph = bt.Metagraph(netuid=netuid, network=network)

    metagraph.sync(subtensor=subtensor)
    logger.info(
        f"Metagraph synced: {metagraph.n} neurons at block {metagraph.block}")

    my_hotkey = wallet.hotkey.ss58_address
    if my_hotkey not in metagraph.hotkeys:
        logger.error(f"Hotkey {my_hotkey} not registered on netuid {netuid}")
        stop_event.set()
        return

    last_weight_block = 0
    weights_rate_limit = get_weights_rate_limit(subtensor, netuid)
    logger.info(f"weights_rate_limit: {weights_rate_limit} blocks")

    async def validator_loop() -> None:
        nonlocal last_weight_block
        while True:
            try:
                metagraph.sync(subtensor=subtensor)
                current_block = subtensor.get_current_block()
                last_heartbeat[0] = time.time()

                blocks_since_last = current_block - last_weight_block
                if blocks_since_last >= weights_rate_limit:
                    logger.info(f"Block {current_block}: evaluating miners")
                    weights_by_uid = await evaluate_and_score(
                        subtensor=subtensor,
                        metagraph=metagraph,
                        netuid=netuid,
                        current_block=current_block,
                    )

                    if weights_by_uid:
                        uids = list(weights_by_uid.keys())
                        weights = [weights_by_uid[uid] for uid in uids]
                        success = subtensor.set_weights(
                            wallet=wallet,
                            netuid=netuid,
                            uids=uids,
                            weights=weights,
                            wait_for_inclusion=True,
                            wait_for_finalization=False,
                        )
                        if success:
                            last_weight_block = current_block
                            logger.info(f"Set weights for {len(uids)} miners")
                        else:
                            logger.warning("Failed to set weights")
                    else:
                        logger.warning("No weights set (empty scores).")
                else:
                    logger.debug(
                        f"Block {current_block}: waiting "
                        f"({blocks_since_last}/{weights_rate_limit} blocks)"
                    )

                await asyncio.sleep(BLOCK_TIME_SECONDS)
            except KeyboardInterrupt:
                logger.info("Validator stopped by user")
                break
            except Exception as exc:
                logger.error(f"Validator loop error: {exc}")
                await asyncio.sleep(BLOCK_TIME_SECONDS)

    try:
        asyncio.run(validator_loop())
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2)


if __name__ == "__main__":
    main()
