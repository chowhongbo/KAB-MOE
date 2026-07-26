import torch
import numpy as np
from typing import List, Dict, Any
from loguru import logger
import torch.nn.functional as F


class LatencyProfiler:
    def __init__(self):
        self.prefill_latencies: List[float] = []
        self.decode_latencies: List[float] = []
        self.batch_size: List[int] = []
        self.seq_len_prefill: List[int] = []
        self.seq_len_decode: List[int] = []
        self.seq_len: List[int] = []
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.enabled = True

    def _forward_pre_hook(self, module: torch.nn.Module, args: tuple):
        if self.enabled:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            module._perf_hook_start_event = start_event

    def _forward_hook(
        self, module: torch.nn.Module, args: tuple, kwargs: Dict[str, Any], output: Any
    ):
        if self.enabled and hasattr(module, "_perf_hook_start_event"):
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()

            latency_ms = module._perf_hook_start_event.elapsed_time(end_event)

            inputs_embeds = None
            if args and isinstance(args[0], torch.Tensor):
                inputs_embeds = args[0]
            elif "inputs_embeds" in kwargs:
                inputs_embeds = kwargs["inputs_embeds"]

            if inputs_embeds is not None:
                seq_len = inputs_embeds.shape[1]
                if seq_len > 1:
                    self.prefill_latencies.append(latency_ms)
                    self.batch_size.append(inputs_embeds.shape[0])
                    self.seq_len_prefill.append(seq_len)
                    self.seq_len_decode.append(0)
                    self.seq_len.append(seq_len)
                else:
                    self.decode_latencies.append(latency_ms)
                    self.seq_len_decode[-1] += 1
                    self.seq_len[-1] += 1

            del module._perf_hook_start_event

    def register(self, model: torch.nn.Module):
        if hasattr(model, "model"):
            target_module = model.model
        else:
            logger.info("No 'model.model' found, try to register on top module.")
            target_module = model

        handle_pre = target_module.register_forward_pre_hook(self._forward_pre_hook)
        handle_post = target_module.register_forward_hook(
            self._forward_hook, with_kwargs=True
        )

        self.handles.extend([handle_pre, handle_post])

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def clear(self):
        self.prefill_latencies = []
        self.decode_latencies = []

    def report(self):
        # --- All setup and helper functions remain identical ---
        WIDTH = 60
        LABEL_WIDTH = 20
        VALUE_WIDTH = 20

        def format_line(label, value, unit=""):
            return f"  {label:<{LABEL_WIDTH}}: {value:>{VALUE_WIDTH}} {unit}"

        # --- We will build a list of lines to be joined later ---
        report_lines = []
        # --- Header construction remains the same ---
        avg_bs_str = (
            f"Avg. BS: {np.mean(self.batch_size):.2f}"
            if hasattr(self, "batch_size") and self.batch_size
            else "Avg. BS: N/A"
        )
        avg_seq_len_str = (
            f"Avg. Seq Len: {np.mean(self.seq_len):.2f}"
            if hasattr(self, "seq_len") and self.seq_len
            else "Avg. Seq Len: N/A"
        )
        header_content = f" {avg_bs_str} | {avg_seq_len_str} "
        title = " Performance Report "
        # --- Append lines to the list instead of logging them ---
        report_lines.append(f"┌{title:─^{WIDTH}}┐")
        report_lines.append(f"│{header_content:^{WIDTH}}│")
        separator = f"├{'─' * WIDTH}┤"
        # --- Handle the case where no data was collected ---
        if not self.prefill_latencies and not self.decode_latencies:
            report_lines.append(f"│{'No performance data was collected.':^{WIDTH}}│")
            report_lines.append(f"└{'─' * WIDTH}┘")
            # --- Single log call and return ---
            logger.info("\n" + "\n".join(report_lines))
            return
        # --- Prefill Section ---
        if self.prefill_latencies:
            report_lines.append(separator)
            avg_prefill_len = np.mean(self.seq_len_prefill)
            prefill_title = f" Prefill (Avg. Seq Len: {avg_prefill_len:.2f}, TTFT) "
            report_lines.append(f"│{prefill_title:^{WIDTH}}│")
            report_lines.append(f"│{' ' * WIDTH}│")
            avg_prefill = np.mean(self.prefill_latencies)
            std_prefill = np.std(self.prefill_latencies)
            report_lines.append(
                f"│{format_line('Number of calls', len(self.prefill_latencies)):<{WIDTH}}│"
            )
            report_lines.append(
                f"│{format_line('Avg. TTFT', f'{avg_prefill:.2f}', 'ms'):<{WIDTH}}│"
            )
            report_lines.append(
                f"│{format_line('Std. TTFT', f'{std_prefill:.2f}', 'ms'):<{WIDTH}}│"
            )
        else:
            report_lines.append(separator)
            report_lines.append(f"│{'Prefill: No data collected.':^{WIDTH}}│")
        # --- Decode Section ---
        if self.decode_latencies:
            report_lines.append(separator)
            avg_decode_len = np.mean(self.seq_len_decode)
            decode_title = f" Decode (Avg. Seq Len: {avg_decode_len:.2f}, TPOT) "
            report_lines.append(f"│{decode_title:^{WIDTH}}│")
            report_lines.append(f"│{' ' * WIDTH}│")
            avg_decode = np.mean(self.decode_latencies)
            std_decode = np.std(self.decode_latencies)
            report_lines.append(
                f"│{format_line('Number of calls', len(self.decode_latencies)):<{WIDTH}}│"
            )
            report_lines.append(
                f"│{format_line('Avg. TPOT', f'{avg_decode:.2f}', 'ms/token'):<{WIDTH}}│"
            )
            report_lines.append(
                f"│{format_line('Std. TPOT', f'{std_decode:.2f}', 'ms/token'):<{WIDTH}}│"
            )
        else:
            report_lines.append(separator)
            report_lines.append(f"│{'Decode: No data collected.':^{WIDTH}}│")
        # --- Footer ---
        report_lines.append(f"└{'─' * WIDTH}┘")
        # --- THE ONLY CHANGE: Join all lines and make a single logger call ---
        # The initial "\n" is preserved from your original code's first logger call.
        final_report_string = "\n" + "\n".join(report_lines)
        logger.info(final_report_string)


def apply_tensor_scale(A, topk_ids, topk_weight, mask):
    """
    Multiply topk_weight[i, j] by A[topk_ids[i, j]] only on rows where mask[i, 0] is True.
    Shapes:
      A: [64]
      topk_ids: [N, 6]
      topk_weight: [N, 6]
      mask: [N, 1] (bool)
    """
    device = topk_weight.device
    A = A.to(device=device, dtype=topk_weight.dtype)
    scale = A[topk_ids]
    mask = mask.expand_as(topk_weight)
    prod = topk_weight * scale
    topk_weight[mask] = prod[mask]
    return topk_weight


def apply_scaler_scale(A, topk_weight, mask):
    """
    Shapes:
      A: float
      topk_weight: [N, 6]
      mask: [N, 1] (bool)
    """
    mask = mask.expand_as(topk_weight)
    prod = topk_weight * A
    topk_weight[mask] = prod[mask]
    return topk_weight


def threshold_and_mask(
    new_topk_weight, topk_weight, topk_ids, mask, threshold: float, max_ids: int
):
    """
    Args:
      new_weight: [N, 6], float tensor
      topk_weight: [N, 6], float tensor
      topk_ids:    [N, 6], long tensor
      mask:   [N, 1], bool tensor (True -> row is text)
      threshold:   float
      max_ids:     int (max id value)

    Returns:
      topk_weight, topk_ids (same shapes as inputs)
    """
    device = topk_weight.device
    dtype = topk_weight.dtype
    mask = mask.expand_as(topk_weight)
    thr = torch.tensor(threshold, device=device, dtype=dtype)
    to_zero = mask & (new_topk_weight < thr)
    topk_weight.masked_fill_(to_zero, 0.0)
    topk_ids.masked_fill_(to_zero, max_ids)

    return topk_weight, topk_ids


def create_conditional_expert_mask(
    top_k_index: torch.Tensor, top_k_weights: torch.Tensor, num_experts: int
) -> torch.Tensor:
    N, k = top_k_index.shape
    M = num_experts
    values_to_scatter = (top_k_weights != 0).int().unsqueeze(-1)
    index_for_scatter = top_k_index.unsqueeze(-1)
    target_tensor = torch.zeros(N, k, M, dtype=torch.int, device=top_k_index.device)
    target_tensor.scatter_(2, index_for_scatter, values_to_scatter)
    return target_tensor.permute(2, 1, 0)
