import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLVisionConfig

from typing import Optional, List, Union, Tuple
# 导入 Qwen2.5-VL 相关的 VisionBlock, VisionRotaryEmbedding, PatchEmbed 等组件
# 确保你能访问到 Qwen2.5-VL 库中用于计算ROPE和处理Patch Embedding的内部工具函数或类

class CustomQwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config: Qwen2_5_VLVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        # 保持原始 __init__ 逻辑，确保所有子模块都已初始化

    # def rot_pos_emb(self, grid_thw, patch_indices):
    #     pos_ids = []
    #     for t, h, w in grid_thw:
    #         hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
    #         hpos_ids = hpos_ids.reshape(
    #             h // self.spatial_merge_size,
    #             self.spatial_merge_size,
    #             w // self.spatial_merge_size,
    #             self.spatial_merge_size,
    #         )
    #         hpos_ids = hpos_ids.permute(0, 2, 1, 3)
    #         hpos_ids = hpos_ids.flatten()

    #         wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
    #         wpos_ids = wpos_ids.reshape(
    #             h // self.spatial_merge_size,
    #             self.spatial_merge_size,
    #             w // self.spatial_merge_size,
    #             self.spatial_merge_size,
    #         )
    #         wpos_ids = wpos_ids.permute(0, 2, 1, 3)
    #         wpos_ids = wpos_ids.flatten()
    #         pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    #     pos_ids = torch.cat(pos_ids, dim=0)
    #     max_grid_size = grid_thw[:, 1:].max()

    #     selected_qwen_pos_ids = pos_ids[patch_indices]
    #     rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
    #     rotary_pos_emb = rotary_pos_emb_full[selected_qwen_pos_ids].flatten(1) 
    #     return rotary_pos_emb



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
        pixel_values: torch.Tensor, # 这是传入的当前子集 patch 的数据
        grid_thw: Optional[torch.LongTensor] = None, # 原始的 image_grid_thw
        is_foreground_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        hidden_states = pixel_values

        # 0. 初始 Patch Embedding 和 RoPE 计算
        # hidden_states_original_patch_order: [总原始 Patch 数, D]
        hidden_states_original_patch_order = self.patch_embed(hidden_states)
        # rotary_pos_emb_original_patch_order: [总原始 Patch 数, RoPE_dim]
        rotary_pos_emb_original_patch_order = self.rot_pos_emb(grid_thw) 

        total_original_patches = hidden_states_original_patch_order.shape[0]
        # `spatial_merge_unit` 表示一个逻辑 Patch 包含多少个原始 Patch
        total_logical_patches = total_original_patches // self.spatial_merge_unit

        # 1. 准备逻辑 Patch 级别的掩码
        # 将原始 Patch 级别的掩码转换为逻辑 Patch 级别
        # 如果一个逻辑 Patch 中的任何子 Patch 属于前景，则整个逻辑 Patch 被认为是前景
        is_foreground_mask_logical = is_foreground_mask.reshape(total_logical_patches, self.spatial_merge_unit).any(dim=1)
        is_background_mask_logical = ~is_foreground_mask_logical # 背景掩码是前景掩码的反转

        # 2. 获取前景和背景各自的窗口索引和 cu_seqlens
        fg_window_index, fg_cu_window_seqlens, fg_original_logical_patch_ids_window_order = \
            self.get_window_index_masked(grid_thw, is_foreground_mask_logical)
        
        bg_window_index, bg_cu_window_seqlens, bg_original_logical_patch_ids_window_order = \
            self.get_window_index_masked(grid_thw, is_background_mask_logical)

        # 3. 准备每个流的 `hidden_states` 和 `position_embeddings`
        # 将原始 `hidden_states` 和 `rotary_pos_emb` 重塑为逻辑单元视图
        hidden_states_logical_unit_view = hidden_states_original_patch_order.reshape(total_logical_patches, self.spatial_merge_unit, -1)
        rotary_pos_emb_logical_unit_view = rotary_pos_emb_original_patch_order.reshape(total_logical_patches, self.spatial_merge_unit, -1)

        # 提取前景和背景的特征和 RoPE，并按照其新的窗口化顺序进行重排
        fg_hidden_states_reordered = hidden_states_logical_unit_view[fg_window_index].reshape(-1, self.config.hidden_size)
        bg_hidden_states_reordered = hidden_states_logical_unit_view[bg_window_index].reshape(-1, self.config.hidden_size)
        
        # 针对 RoPE，Qwen 是 (emb.cos(), emb.sin()) 对
        # 确保 rotary_pos_emb_original_patch_order 是 head_dim 维度的，因为它会 `cat((rotary_pos_emb, rotary_pos_emb), dim=-1)`
        # 所以这里的 `rotary_pos_emb_original_patch_order.shape[-1]` 应该就是 head_dim
        
        # 假设 self.rot_pos_emb 返回的维度就是 head_dim
        fg_rotary_pos_emb_reordered_single_dim = rotary_pos_emb_logical_unit_view[fg_window_index].reshape(-1, self.config.hidden_size // (self.config.num_heads * 2)) # 假设 RoPE 的维度是 hidden_size // (num_heads * 2)
        bg_rotary_pos_emb_reordered_single_dim = rotary_pos_emb_logical_unit_view[bg_window_index].reshape(-1, self.config.hidden_size // (self.config.num_heads * 2))

        # 确保 RoPE 维度与模型期望一致 (通常是 head_dim)
        fg_position_embeddings = (
            torch.cat((fg_rotary_pos_emb_reordered_single_dim, fg_rotary_pos_emb_reordered_single_dim), dim=-1).cos(),
            torch.cat((fg_rotary_pos_emb_reordered_single_dim, fg_rotary_pos_emb_reordered_single_dim), dim=-1).sin()
        )
        bg_position_embeddings = (
            torch.cat((bg_rotary_pos_emb_reordered_single_dim, bg_rotary_pos_emb_reordered_single_dim), dim=-1).cos(),
            torch.cat((bg_rotary_pos_emb_reordered_single_dim, bg_rotary_pos_emb_reordered_single_dim), dim=-1).sin()
        )
        
        # 4. Transformer 块处理循环 (前景和背景完全独立)
        current_fg_hidden_states = fg_hidden_states_reordered
        current_bg_hidden_states = bg_hidden_states_reordered

       # --- 重要修改：如果某个流为空，跳过其 Transformer 块处理 ---
        num_fg_patches = current_fg_hidden_states.shape[0]
        num_bg_patches = current_bg_hidden_states.shape[0]

        # 只有当存在前景 Patch 时才处理前景流
        if num_fg_patches > 0:
            for layer_num, blk in enumerate(self.blocks):
                current_fg_hidden_states = blk(
                    current_fg_hidden_states,
                    cu_seqlens=fg_cu_window_seqlens,
                    position_embeddings=fg_position_embeddings
                )
        # 否则，current_fg_hidden_states 保持为空，这对于后续步骤是没问题的

        # 只有当存在背景 Patch 时才处理背景流
        if num_bg_patches > 0:
            for layer_num, blk in enumerate(self.blocks):
                current_bg_hidden_states = blk(
                    current_bg_hidden_states,
                    cu_seqlens=bg_cu_window_seqlens,
                    position_embeddings=bg_position_embeddings
                )
        # 否则，current_bg_hidden_states 保持为空

        # --- 应用 merger ---
        # merger 也需要能优雅地处理空输入，或者我们确保它不会接收到空输入。
        # 它的输出形状将是 (逻辑 Patch 数量, out_hidden_size)
        fg_features_after_merger = torch.empty(0, self.config.out_hidden_size, device=pixel_values.device, dtype=pixel_values.dtype)
        if num_fg_patches > 0: # 仅当存在前景 Patch 时才调用 merger
            fg_features_after_merger = self.merger(current_fg_hidden_states)
        
        bg_features_after_merger = torch.empty(0, self.config.out_hidden_size, device=pixel_values.device, dtype=pixel_values.dtype)
        if num_bg_patches > 0: # 仅当存在背景 Patch 时才调用 merger
            bg_features_after_merger = self.merger(current_bg_hidden_states)

        # 重排合并后的逻辑 Patch 特征到原始逻辑 Patch ID 顺序
        # 这些索引是基于逻辑 Patch 的。
        # 如果一个流为空，它的 window_index 也会为空，argsort 将返回一个空张量。
        fg_reverse_logical_patch_order = torch.argsort(fg_window_index)
        bg_reverse_logical_patch_order = torch.argsort(bg_window_index)

        # 仅当存在特征需要重排时才进行重排
        fg_features_final_ordered = fg_features_after_merger
        if fg_features_final_ordered.shape[0] > 0:
            fg_features_final_ordered = fg_features_after_merger[fg_reverse_logical_patch_order]
        
        bg_features_final_ordered = bg_features_after_merger
        if bg_features_final_ordered.shape[0] > 0:
            bg_features_final_ordered = bg_features_after_merger[bg_reverse_logical_patch_order]

        # 最终拼接
        final_result_bg_then_fg = torch.cat((bg_features_final_ordered, fg_features_final_ordered), dim=0)

        return final_result_bg_then_fg
