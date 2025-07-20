import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

from typing import Optional, List, Union, Tuple

import torch.nn.functional as F
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionConfig

# 导入 Qwen2.5-VL 相关的 VisionBlock, VisionRotaryEmbedding, PatchEmbed 等组件
# 确保你能访问到 Qwen2.5-VL 库中用于计算ROPE和处理Patch Embedding的内部工具函数或类

class CustomQwen2_VisionTransformerPretrainedModel(Qwen2VisionTransformerPretrainedModel):
    def __init__(self, config: Qwen2VLVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        # 保持原始 __init__ 逻辑，确保所有子模块都已初始化

        self.mask_windows_granularity_size = 4

    def _apply_granularity_to_mask(
        self,
        is_foreground_mask: torch.Tensor,
        grid_thw: torch.LongTensor,
    ) -> torch.Tensor:
        """
        根据设定的粒度，统一一个区域块的mask。
        如果一个块内有任何一个前景patch，则整个块都视为前景。
        """
        # 假设处理单张图片
        if grid_thw.shape[0] != 1:
            raise NotImplementedError("Granularity masking supports single image processing only.")
        
        # is_foreground_mask 是一维的 (num_patches,)
        # grid_thw 的形状是 (1, 3)，内容是 [T, H, W]，T通常是1
        _, grid_h, grid_w = grid_thw[0]
        
        # 将一维mask变回二维图像形状
        mask_grid = is_foreground_mask.view(grid_h.item(), grid_w.item())
        
        granularity = self.mask_granularity_size
        pad_h = (granularity - grid_h % granularity) % granularity
        pad_w = (granularity - grid_w % granularity) % granularity
        
        # 对mask进行padding，使其尺寸能被粒度整除
        mask_padded = F.pad(mask_grid.float(), (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # 使用reshape和any实现高效的块内投票
        ph, pw = mask_padded.shape
        # 将mask重塑成 (H/G, G, W/G, G) 的形式
        blocks = mask_padded.view(ph // granularity, granularity, pw // granularity, granularity)
        
        # 在每个块(G,G)内，只要有1(前景)，结果就为1
        block_votes = blocks.any(dim=3).any(dim=1).float() # 形状 (H/G, W/G)
        
        # 将投票结果放大回padding后的大小
        final_mask_padded = torch.repeat_interleave(
            torch.repeat_interleave(block_votes, granularity, dim=1),
            granularity,
            dim=0
        )
        
        # 裁剪回原始的grid大小，并转换为布尔型
        final_mask_grid = final_mask_padded[:grid_h, :grid_w].bool()
        
        # 展平回一维
        return final_mask_grid.flatten()
    

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.LongTensor] = None,
        is_foreground_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        # 1. Physical Patch Operations (using parent's attributes directly)
        hidden_states = self.patch_embed(pixel_values)

        device = hidden_states.device
        is_foreground_mask = is_foreground_mask.to(device)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)        
        position_embeddings = (emb.cos(), emb.sin())

        self.num_patches_per_logical = self.spatial_merge_size * self.spatial_merge_size
        # 2. Derive the final logical mask with window granularity
        # Step 2.1: Get initial logical mask from the per-physical-patch mask
        initial_logical_mask = is_foreground_mask.view(
            -1, self.num_patches_per_logical
        ).any(dim=1)

        # Step 2.2: Apply windowed voting on the logical grid
        _, grid_h_physical, grid_w_physical = grid_thw[0]
        grid_h_logical = grid_h_physical.item() // self.spatial_merge_size
        grid_w_logical = grid_w_physical.item() // self.spatial_merge_size
        
        logical_mask_grid = initial_logical_mask.view(grid_h_logical, grid_w_logical)
        
        granularity = self.mask_windows_granularity_size
        pad_h = (granularity - grid_h_logical % granularity) % granularity
        pad_w = (granularity - grid_w_logical % granularity) % granularity
        
        mask_padded = F.pad(logical_mask_grid.float(), (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        ph, pw = mask_padded.shape
        blocks = mask_padded.view(ph // granularity, granularity, pw // granularity, granularity)
        
        # If any logical token in a window is foreground, the whole window is foreground
        block_votes = blocks.any(dim=3).any(dim=1).float()
        
        final_mask_padded = torch.repeat_interleave(
            torch.repeat_interleave(block_votes, granularity, dim=1), granularity, dim=0
        )
        
        final_mask_grid = final_mask_padded[:grid_h_logical, :grid_w_logical].bool()
        final_logical_is_foreground_mask = final_mask_grid.flatten()
        
        # 3. Get reorder indices for logical tokens based on the final mask
        logical_bg_indices = (~final_logical_is_foreground_mask).nonzero(as_tuple=True)[0]
        logical_fg_indices = final_logical_is_foreground_mask.nonzero(as_tuple=True)[0]
        logical_reorder_indices = torch.cat([logical_bg_indices, logical_fg_indices])

        # 4. Calculate physical patch reorder indices from logical ones
        num_logical_tokens = final_logical_is_foreground_mask.shape[0]
        physical_reorder_indices = torch.repeat_interleave(
            logical_reorder_indices, self.num_patches_per_logical
        ) * self.num_patches_per_logical
        offsets = torch.arange(self.num_patches_per_logical, device=device).repeat(num_logical_tokens)
        physical_reorder_indices += offsets

        # 5. Reorder physical patches and their positional embeddings
        hidden_states = hidden_states[physical_reorder_indices]
        pos_emb_cos = position_embeddings[0][physical_reorder_indices]
        pos_emb_sin = position_embeddings[1][physical_reorder_indices]
        reordered_position_embeddings = (pos_emb_cos, pos_emb_sin)
        
        # 6. Process the reordered stream through Transformer blocks
        num_bg_physical = len(logical_bg_indices) * self.num_patches_per_logical
        num_total_physical = hidden_states.shape[0]
        cu_seqlens = torch.tensor([0, num_bg_physical, num_total_physical], device=device, dtype=torch.int32)
        
        # Loop through the parent's `self.blocks`
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Assuming `_gradient_checkpointing_func` is available or defined
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, None, reordered_position_embeddings
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=reordered_position_embeddings
                )
        
        # 7. Merge physical patches to logical tokens (using parent's `self.merger`)
        merged_hidden_states = self.merger(hidden_states)
        
        # 8. Return sorted logical tokens and the logical-level reorder indices
        return merged_hidden_states, logical_reorder_indices