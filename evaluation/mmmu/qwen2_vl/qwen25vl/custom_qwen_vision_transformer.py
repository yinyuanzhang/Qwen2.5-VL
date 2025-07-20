import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLVisionConfig

from typing import Optional, List, Union, Tuple

import torch.nn.functional as F

# 导入 Qwen2.5-VL 相关的 VisionBlock, VisionRotaryEmbedding, PatchEmbed 等组件
# 确保你能访问到 Qwen2.5-VL 库中用于计算ROPE和处理Patch Embedding的内部工具函数或类

class CustomQwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config: Qwen2_5_VLVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        # 保持原始 __init__ 逻辑，确保所有子模块都已初始化

    def get_window_index_masked(self, grid_thw: torch.Tensor, patch_mask_logical: torch.Tensor):
        # patch_mask_logical: [total_logical_patches] 的布尔掩码，指示哪些 Patch 属于当前组 (前景或背景)
        
        window_index_filtered: List[torch.Tensor] = [] # 过滤后的窗口化 Patch 索引 (指向原始逻辑 Patch ID)
        cu_window_seqlens_filtered: List[int] = [0] # 过滤后的窗口累积序列长度
        # 记录每个被选中的 Patch 在其原始的“全局逻辑 Patch 序列”中的 ID，顺序是窗口化重排后的
        original_logical_patch_ids_in_window_order: List[torch.Tensor] = [] 

        current_logical_patch_id_offset = 0 # 当前帧在总逻辑 Patch 序列中的起始偏移

        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            # `index` 包含当前帧内逻辑 Patch 的原始相对 ID (0 到 llm_grid_h*llm_grid_w - 1)
            index = torch.arange(llm_grid_h * llm_grid_w, device=patch_mask_logical.device).reshape(1, llm_grid_h, llm_grid_w)
            index = index.repeat(grid_t, 1, 1)

            # 获取当前帧对应 Patch 的逻辑掩码切片
            frame_logical_patch_count = grid_t * llm_grid_h * llm_grid_w
            current_frame_mask_logical = patch_mask_logical[current_logical_patch_id_offset : current_logical_patch_id_offset + frame_logical_patch_count].reshape(grid_t, llm_grid_h, llm_grid_w)
            
            # 填充 (与 Qwen 原始逻辑相同)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            
            import torch.nn.functional as F
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -1)
            mask_padded = F.pad(current_frame_mask_logical.float(), (0, pad_w, 0, pad_h), "constant", 0).bool()

            # 重排为窗口格式
            index_padded = index_padded.reshape(
                grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
            ).permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
            )
            mask_padded = mask_padded.reshape(
                grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
            ).permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
            )
            
            # 展平以便过滤
            index_padded_flat = index_padded.reshape(-1)
            mask_padded_flat = mask_padded.reshape(-1)

            # --- 核心：根据掩码过滤 Patch ---
            # 只有非填充的 Patch 且符合当前组的掩码才被选中
            valid_patch_mask_in_padded_order = (index_padded_flat != -1) & mask_padded_flat
            
            # 选中 Patch 在当前帧内的原始逻辑 ID (重排后)
            index_new_filtered = index_padded_flat[valid_patch_mask_in_padded_order]
            
            # 每个窗口中有效 Patch 的数量
            seqlens_per_window = valid_patch_mask_in_padded_order.reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
            ).sum([2, 3]).reshape(-1)

            # --- 结果收集 ---
            # 加上全局偏移，得到全局逻辑 Patch ID
            global_logical_ids_for_filtered = index_new_filtered + current_logical_patch_id_offset
            window_index_filtered.append(global_logical_ids_for_filtered)
            original_logical_patch_ids_in_window_order.append(global_logical_ids_for_filtered) # 保存原始 ID，顺序是窗口化后的

            # 更新 cu_seqlens
            cu_seqlens_tmp = seqlens_per_window.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens_filtered[-1]
            cu_window_seqlens_filtered.extend(cu_seqlens_tmp.tolist())
            
            current_logical_patch_id_offset += frame_logical_patch_count

        window_index_filtered = torch.cat(window_index_filtered, dim=0)
        original_logical_patch_ids_in_window_order = torch.cat(original_logical_patch_ids_in_window_order, dim=0)

        cu_window_seqlens_filtered = torch.tensor(
            cu_window_seqlens_filtered,
            device=patch_mask_logical.device,
            dtype=torch.int32,
        )
        cu_window_seqlens_filtered = torch.unique_consecutive(cu_window_seqlens_filtered)

        return window_index_filtered, cu_window_seqlens_filtered, original_logical_patch_ids_in_window_order



    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.LongTensor] = None,
        is_foreground_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # 原始输入 pixel_values 已经是 patchified 的 hidden_states
        hidden_states = pixel_values

        # 1. Patch Embedding (如果需要，根据你的ViT输入)
        # 在你的场景下，pixel_values 已经是 patch 嵌入，所以这步可能叫别的名字或不需要
        hidden_states = self.patch_embed(hidden_states)

        # 2. 计算全局RoPE (保持不变，逻辑正确)
        # rotary_pos_emb 是按照原始逻辑顺序排列的
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        
        # 3. 获取窗口索引和窗口级别的cu_seqlens (保持不变，逻辑正确)
        window_index, cu_window_seqlens_all_list = self.get_window_index(grid_thw)
        cu_window_seqlens_all = torch.tensor(
            cu_window_seqlens_all_list,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        cu_window_seqlens_all = torch.unique_consecutive(cu_window_seqlens_all)
        seq_len = hidden_states.size(0)

        # 4. 根据 window_index 重排 hidden_states 和 rotary_pos_emb (保持不变，逻辑正确)
        hidden_states_reshaped = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states_window_ordered = hidden_states_reshaped[window_index, :, :].reshape(seq_len, -1)

        rotary_pos_emb_reshaped = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb_window_ordered = rotary_pos_emb_reshaped[window_index, :, :].reshape(seq_len, -1)

        emb = torch.cat((rotary_pos_emb_window_ordered, rotary_pos_emb_window_ordered), dim=-1)
        position_embeddings_window_ordered = (emb.cos(), emb.sin())

        # 5. 对齐Mask (保持不变，逻辑正确)
        # is_foreground_mask 已经是 per-patch 的 mask
        is_foreground_mask_logical_patch_level = is_foreground_mask.reshape(-1, self.spatial_merge_unit).any(dim=-1)
        is_foreground_mask_window_ordered = is_foreground_mask_logical_patch_level[window_index]

        # --- 从这里开始是核心修改逻辑 ---

        # 6. 确定每个窗口的归属 (前景 or 背景)
        cu_window_seqlens_logical = cu_window_seqlens_all // self.spatial_merge_unit
        num_total_windows = cu_window_seqlens_logical.shape[0] - 1
        is_window_foreground = torch.zeros(num_total_windows, dtype=torch.bool, device=hidden_states.device)

        for i in range(num_total_windows):
            start, end = cu_window_seqlens_logical[i], cu_window_seqlens_logical[i+1]
            if is_foreground_mask_window_ordered[start:end].any():
                is_window_foreground[i] = True

        # 7. 分离前景和背景数据，并构建各自的两套 cu_seqlens
        fg_indices, bg_indices = [], []
        fg_window_patch_counts, bg_window_patch_counts = [], []
        
        for i in range(num_total_windows):
            start, end = cu_window_seqlens_all[i], cu_window_seqlens_all[i+1]
            window_patch_indices = torch.arange(start, end, device=hidden_states.device)
            num_patches_in_window = len(window_patch_indices)
            
            if is_window_foreground[i]:
                fg_indices.append(window_patch_indices)
                fg_window_patch_counts.append(num_patches_in_window)
            else:
                bg_indices.append(window_patch_indices)
                bg_window_patch_counts.append(num_patches_in_window)

        # 合并索引
        foreground_indices = torch.cat(fg_indices) if fg_indices else torch.empty(0, dtype=torch.long, device=hidden_states.device)
        background_indices = torch.cat(bg_indices) if bg_indices else torch.empty(0, dtype=torch.long, device=hidden_states.device)
        
        # 提取数据流
        hidden_states_fg = hidden_states_window_ordered[foreground_indices]
        pos_emb_fg = (position_embeddings_window_ordered[0][foreground_indices], position_embeddings_window_ordered[1][foreground_indices])
        
        hidden_states_bg = hidden_states_window_ordered[background_indices]
        pos_emb_bg = (position_embeddings_window_ordered[0][background_indices], position_embeddings_window_ordered[1][background_indices])

        # *** 关键修正：为前景和背景流分别创建 "窗口" 和 "全局" 两种 cu_seqlens ***
        
        # a) 窗口注意力所需的 cu_seqlens (你的原始实现是正确的)
        cu_seqlens_window_fg = torch.tensor([0] + list(torch.tensor(fg_window_patch_counts).cumsum(0)), device=hidden_states.device, dtype=torch.int32)
        cu_seqlens_window_bg = torch.tensor([0] + list(torch.tensor(bg_window_patch_counts).cumsum(0)), device=hidden_states.device, dtype=torch.int32)

        # b) 全局注意力所需的 cu_seqlens
        # 对于单张图片，全局cu_seqlens只包含0和总长度
        num_fg_patches = hidden_states_fg.shape[0]
        num_bg_patches = hidden_states_bg.shape[0]
        cu_seqlens_global_fg = torch.tensor([0, num_fg_patches], device=hidden_states.device, dtype=torch.int32)
        cu_seqlens_global_bg = torch.tensor([0, num_bg_patches], device=hidden_states.device, dtype=torch.int32)
        
        # 8. 分别通过 Transformer Blocks (核心修正)
        for layer_num, blk in enumerate(self.blocks):
            # *** 根据层类型选择正确的 cu_seqlens ***
            if layer_num in self.fullatt_block_indexes:
                # 使用全局 cu_seqlens
                cu_seqlens_fg_now = cu_seqlens_global_fg
                cu_seqlens_bg_now = cu_seqlens_global_bg
            else:
                # 使用窗口 cu_seqlens
                cu_seqlens_fg_now = cu_seqlens_window_fg
                cu_seqlens_bg_now = cu_seqlens_window_bg

            # 处理前景流
            if num_fg_patches > 0:
                if self.gradient_checkpointing and self.training:
                    hidden_states_fg = self._gradient_checkpointing_func(
                        blk.__call__, hidden_states_fg, cu_seqlens_fg_now, None, pos_emb_fg
                    )
                else:
                    hidden_states_fg = blk(hidden_states_fg, cu_seqlens=cu_seqlens_fg_now, position_embeddings=pos_emb_fg)

            # 处理背景流
            if num_bg_patches > 0:
                if self.gradient_checkpointing and self.training:
                    hidden_states_bg = self._gradient_checkpointing_func(
                        blk.__call__, hidden_states_bg, cu_seqlens_bg_now, None, pos_emb_bg
                    )
                else:
                    hidden_states_bg = blk(hidden_states_bg, cu_seqlens=cu_seqlens_bg_now, position_embeddings=pos_emb_bg)

        # 9. 合并前景和背景的 hidden_states (保持不变，逻辑正确)
        hidden_states_window_ordered_processed = torch.empty_like(hidden_states_window_ordered)
        if num_fg_patches > 0:
            hidden_states_window_ordered_processed[foreground_indices] = hidden_states_fg
        if num_bg_patches > 0:
            hidden_states_window_ordered_processed[background_indices] = hidden_states_bg

        # 10. 合并器 (Merger) (保持不变，逻辑正确)
        hidden_states_merged = self.merger(hidden_states_window_ordered_processed)

        # 11. 逆向窗口重排，恢复到原始的 logical patch 顺序 (保持不变，逻辑正确)
        reverse_indices = torch.argsort(window_index)
        hidden_states_final = hidden_states_merged[reverse_indices, :]

        final_is_foreground_mask = torch.zeros_like(is_foreground_mask_logical_patch_level)

        # 遍历逻辑窗口，根据窗口的归属（is_window_foreground）标记最终的mask
        for i in range(num_total_windows):
            start, end = cu_window_seqlens_logical[i], cu_window_seqlens_logical[i+1]
            # 获取原始逻辑索引对应的 patch 在 window_index 中的位置
            original_logical_indices = window_index[start:end] 
            if is_window_foreground[i]:
                final_is_foreground_mask[original_logical_indices] = True
            else:
                final_is_foreground_mask[original_logical_indices] = False
        return hidden_states_final, final_is_foreground_mask