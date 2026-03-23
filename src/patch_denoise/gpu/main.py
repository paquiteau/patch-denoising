"""Main loop for the gpu version of patch-denoise."""

import queue
import threading
import logging

from tqdm.auto import tqdm
import torch
import numpy as np
import cupy as cp
import numba as nb

from .dataloader import PatchDataset
from ..space_time.base import PatchedArray


@nb.njit(parallel=True, cache=True)
def _accumulator(out_acc, out_weights, batch_data, batch_weights, batch_indices):
    """Accumulate resulted denoised data asynchronously.

    Bypass the GIL with Numba
    """
    B, PH, PW, PD, PT = batch_data.shape
    for b in nb.prange(B):
        i, j, k, l = batch_indices[b]
        # In-place addition into the large pinned RAM buffer
        out_acc[i : i + PH, j : j + PW, k : k + PD, l : l + PT] += batch_data[b]
        out_weights[i : i + PH, j : j + PW, k : k + PD, l : l + PT] += batch_weights[b]


def make_denoiser(args, noise_map, **kwargs) -> torch.nn.Module:
    """Create a denoiser model on GPU."""
    from .denoiser import OptimalSVDDenoiser

    print(args)
    denoiser = OptimalSVDDenoiser(
        patch_shape=args.patch_shape,
        recombination=args.recombination,
        loss=args.opt_loss,
        **kwargs,
    )
    return denoiser


@torch.inference_mode()
def main_gpu(args, input_data, mask, noise_map, batch_size=32, **kwargs):
    """Denoise loop for the gpu version of patch-denoise."""
    # Create the Dataset
    out_weights = torch.zeros_like(input_data).pin_memory_()
    out_acc = torch.zeros_like(input_data).pin_memory_()

    # setup dataset and dataloader using pytorch api:
    patch_dataset = PatchDataset(
        input_data,
        patch_shape=args.patch_shape,
        patch_overlap=args.patch_overlap,
        mask=mask,
        noise_map=noise_map,
        mask_threshold=args.mask_threshold,
    )
    loader = torch.utils.data.DataLoader(
        patch_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # No multiprocessing to avoid GPU contention
        pin_memory=True,  # Pin memory for faster CPU-GPU transfer
    )

    # Setup the denoiser model on GPU
    denoiser = make_denoiser(args, noise_map, **kwargs).cuda()

    # Scheduling with double buffering and a worker thread to accumulate results asynchronously
    res_queue = queue.Queue(maxsize=10)  # To hold results from GPU threads
    N_STREAMS = 2
    streams = [torch.cuda.Stream() for _ in range(N_STREAMS)]
    events = [torch.cuda.Event() for _ in range(N_STREAMS)]

    def _worker():
        """Worker thread to accumulate results return from GPU asynchronously."""
        while True:
            item = res_queue.get()
            if item is None:  # Sentinel to stop the thread
                break
            cpu_out, cpu_weights, indices, event = item

            # convert linear indices back to 4D patch indices
            batch_indices = np.array(
                [
                    PatchedArray.linear_to_patch_indices(
                        idx, input_data.shape, args.patch_shape, args.patch_overlap
                    )
                    for idx in indices
                ]
            )
            event.synchronize()  # Wait for GPU to finish
            _accumulator(
                out_acc,
                out_weights,
                cpu_out.numpy(),
                cpu_weights,
                batch_indices,
            )

    worker_thread = threading.Thread(target=_worker)
    worker_thread.start()

    logging.info(
        f"Processing {len(loader.dataset)} patches with batch size {args.batch_size}..."
    )
    for i, (patches, indices) in enumerate(tqdm(loader)):
        slot = i % 2
        stream = streams[slot]
        event = events[slot]

        with torch.cuda.stream(stream):
            # H2D -> Compute -> D2H (All Asynchronous)
            gpu_in = patches.cuda(non_blocking=True)
            gpu_out = denoiser(gpu_in)
            cpu_out = gpu_out.to("cpu", non_blocking=True)

            event.record()  # Mark when this slot is done

        res_queue.put((cpu_out, indices, event))

    res_queue.put(None)
    worker_thread.join()
