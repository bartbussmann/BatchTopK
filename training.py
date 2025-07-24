import torch
import tqdm
from logs import init_wandb, log_wandb, save_checkpoint, save_checkpoint_mp
import multiprocessing as mp
from queue import Empty
import wandb
import json
import os
import time

def train_sae(sae, activation_store, model, cfg):
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)
    
    for i in pbar:
        batch = activation_store.next_batch()
        sae_output = sae(batch)
        log_wandb(sae_output, i, wandb_run)
        if i % cfg["perf_log_freq"]  == 0:
            log_model_performance(wandb_run, i, model, activation_store, sae)

        if i % cfg["checkpoint_freq"] == 0:
            save_checkpoint(wandb_run, sae, cfg, i)

        loss = sae_output["loss"]
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    save_checkpoint(wandb_run, sae, cfg, i)
    

def train_sae_group_separate_wandb(saes, activation_store, model, cfgs):
    def new_wandb_process(config, log_queue, entity, project):
        # Remove W_U and b_U from config before submitting to wandb
        wandb_config = {k: v for k, v in config.items() if k not in ["W_U", "b_U"]}
        run = wandb.init(
            entity=entity, project=project, config=wandb_config, name=config["name"]
        )

        while True:
            try:
                # Wait up to 1 second for new data
                log = log_queue.get(timeout=1)

                # Check for termination signal
                if log == "DONE":
                    break

                # Check if this is a checkpoint signal
                if isinstance(log, dict) and log.get("checkpoint"):
                    # Create and log artifact
                    artifact = wandb.Artifact(
                        name=f"{config['name']}_{log['step']}",
                        type="model",
                        description=f"Model checkpoint at step {log['step']}",
                    )
                    save_dir = log["save_dir"]
                    artifact.add_file(os.path.join(save_dir, "sae.pt"))
                    artifact.add_file(os.path.join(save_dir, "config.json"))
                    run.log_artifact(artifact)
                else:
                    # Log regular metrics
                    wandb.log(log)
            except Empty:
                continue

        wandb.finish()

    num_batches = int(cfgs[0]["num_tokens"] // cfgs[0]["batch_size"])
    print(f"Number of batches: {num_batches}")
    optimizers = [
        torch.optim.Adam(
            sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
        )
        for sae, cfg in zip(saes, cfgs)
    ]
    pbar = tqdm.trange(num_batches)

    # Initialize wandb processes and queues
    wandb_processes = []
    log_queues = []

    for i, cfg in enumerate(cfgs):
        log_queue = mp.Queue()
        log_queues.append(log_queue)
        wandb_config = cfg
        wandb_process = mp.Process(
            target=new_wandb_process,
            args=(
                wandb_config,
                log_queue,
                cfg.get("wandb_entity", ""),
                cfg["wandb_project"],
            ),
        )
        wandb_process.start()

        # Wait a bit 
        time.sleep(3)
        wandb_processes.append(wandb_process)

    for i in pbar:
        batch = activation_store.next_batch()

        for idx, (sae, cfg, optimizer) in enumerate(
            zip(saes, cfgs, optimizers)
        ):
            sae_output = sae(batch)
            loss = sae_output["loss"]

            # Log metrics to appropriate wandb process
            log_dict = {
                k: (
                    v.item()
                    if isinstance(v, torch.Tensor) and v.dim() == 0
                    else v
                )
                for k, v in sae_output.items()
                if isinstance(v, (int, float))
                or (isinstance(v, torch.Tensor) and v.dim() == 0)
            }
            log_queues[idx].put(log_dict)

            if i != 0 and i % cfg["checkpoint_freq"] == 0:
                # Save checkpoint and send artifact info to wandb process
                save_dir, _, _ = save_checkpoint_mp(sae, cfg, i)
                log_queues[idx].put(
                    {"checkpoint": True, "step": i, "save_dir": save_dir}
                )

            pbar.set_postfix(
                {
                    f"Loss_{idx}": f"{loss.item():.4f}",
                    f"L0_{idx}": f"{sae_output['l0_norm']:.4f}",
                    f"L2_{idx}": f"{sae_output['l2_loss']:.4f}",
                    f"L1_{idx}": f"{sae_output['l1_loss']:.4f}",
                }
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                sae.parameters(), cfg["max_grad_norm"]
            )

            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

    # Final checkpoints
    for idx, (sae, cfg) in enumerate(zip(saes, cfgs)):
        save_dir, _, _ = save_checkpoint_mp(sae, cfg, i)
        log_queues[idx].put(
            {"checkpoint": True, "step": i, "save_dir": save_dir}
        )

    # Clean up wandb processes
    for queue in log_queues:
        queue.put("DONE")
    for process in wandb_processes:
        process.join()


