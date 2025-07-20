import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import QWEN2_5_VL_INPUTS_DOCSTRING, Qwen2_5_VLCausalLMOutputWithPast

# 确保导入所有 Qwen2.5-VL 原始模型依赖，如 Qwen2_5_VLModel, Qwen2_5_VLCausalLMOutputWithPast 等
from typing import Optional, List, Union, Tuple

# 从第一步导入我们自定义的 Vision Transformer 模型
from .custom_qwen_vision_transformer import CustomQwen2_5_VisionTransformerPretrainedModel 

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from torch.nn import CrossEntropyLoss


_CONFIG_FOR_DOC = "Qwen2_5_VLConfig"


class CustomQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.visual = CustomQwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)

    # # ===========================================================================
    # # 覆盖 get_rope_index 方法
    # # ===========================================================================
    # def get_rope_index(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     second_per_grid_ts: Optional[torch.Tensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     # 新增一个参数，用来传递 ViT 输出的前景/背景分块信息
    #     # 假设你能够从 CustomQwen2_5_VisionTransformerPretrainedModel 中获取
    #     # 原始的逻辑 Patch IDs (背景在前，前景在后)
    #     # 例如，可以是一个列表或张量，记录了每个图像特征块在原始图像中的逻辑位置ID
    #     # 这里我假设你需要一个参数来传递背景和前景各自的逻辑 Patch 数量
    #     bg_logical_patch_counts_per_image: Optional[List[int]] = None,
    #     fg_logical_patch_counts_per_image: Optional[List[int]] = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:

    #     # 核心逻辑：这里需要根据你 ViT 输出的 "背景在前，前景在后" 的顺序
    #     # 来重新构建 position_ids。

    #     # 1. 首先，像原始 get_rope_index 一样处理纯文本部分
    #     # 2. 当遇到图像/视频token时，不再使用原始的 image_grid_thw/video_grid_thw 顺序
    #     #    而是根据 bg_logical_patch_counts_per_image 和 fg_logical_patch_counts_per_image
    #     #    来分配位置 ID。

    #     spatial_merge_size = self.config.vision_config.spatial_merge_size
    #     image_token_id = self.config.image_token_id
    #     video_token_id = self.config.video_token_id
    #     vision_start_token_id = self.config.vision_start_token_id
    #     mrope_position_deltas = []

    #     if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
    #         total_input_ids = input_ids
    #         if attention_mask is None:
    #             attention_mask = torch.ones_like(total_input_ids)
            
    #         # position_ids 初始化
    #         position_ids = torch.ones(
    #             3,
    #             input_ids.shape[0],
    #             input_ids.shape[1],
    #             dtype=input_ids.dtype,
    #             device=input_ids.device,
    #         )
            
    #         # current_image_idx 和 current_video_idx 用于跟踪原始 image_grid_thw/video_grid_thw
    #         # 但现在我们需要一个新的索引来跟踪 bg/fg 计数
    #         current_image_original_grid_idx = 0 
    #         current_video_original_grid_idx = 0
            
    #         # 用于跟踪当前图像/视频的背景和前景的已处理 Patch 数量
    #         current_bg_processed_count = 0
    #         current_fg_processed_count = 0

    #         attention_mask = attention_mask.to(total_input_ids.device)

    #         for i, input_ids_batch in enumerate(total_input_ids): # 遍历每个 batch
    #             input_ids_filtered = input_ids_batch[attention_mask[i] == 1] # 过滤掉填充
    #             input_tokens_list = input_ids_filtered.tolist()

    #             llm_pos_ids_list: list = []
    #             current_sequence_start_idx = 0 # 当前 batch 中，当前处理段的起始全局索引

    #             # 获取当前 batch 中前景/背景的 Patch 数量
    #             num_bg_patches_current_image = bg_logical_patch_counts_per_image[i] if bg_logical_patch_counts_per_image else 0
    #             num_fg_patches_current_image = fg_logical_patch_counts_per_image[i] if fg_logical_patch_counts_per_image else 0
                
    #             # 计算当前图像/视频的总逻辑 Patch 数
    #             # 这里假设每个图像/视频的特征块是连续放置的
    #             total_visual_logical_patches_current_item = num_bg_patches_current_image + num_fg_patches_current_image

    #             # 找到视觉token的起始位置 (例如 <image> 或 <video>)
    #             vision_start_token_pos_in_input_ids = -1
    #             if image_token_id in input_tokens_list:
    #                 vision_start_token_pos_in_input_ids = input_tokens_list.index(vision_token_id) # 或 image_token_id

    #             if vision_start_token_pos_in_input_ids != -1:
    #                 # 处理视觉token之前的文本部分
    #                 text_len_before_vision = vision_start_token_pos_in_input_ids
    #                 if text_len_before_vision > 0:
    #                     st_idx = 0
    #                     llm_pos_ids_list.append(torch.arange(text_len_before_vision).view(1, -1).expand(3, -1) + st_idx)
    #                     current_sequence_start_idx += text_len_before_vision

    #                 # ==========================================================
    #                 # 核心逻辑：根据背景在前，前景在后的顺序生成位置ID
    #                 # ==========================================================

    #                 # 假设 bg_logical_patch_counts_per_image 和 fg_logical_patch_counts_per_image
    #                 # 包含了你 ViT 实际输出的背景和前景逻辑 Patch 的数量
                    
    #                 # 1. 处理背景 Patch 的位置ID
    #                 if num_bg_patches_current_image > 0:
    #                     # 找到当前图像/视频在 image_grid_thw 或 video_grid_thw 中的对应信息
    #                     # 这里你需要一个方法来根据 batch index i 找到对应的 grid_thw
    #                     # 假设我们能拿到当前图像/视频的原始 T, H, W
                        
    #                     # 为了简化，我们假设 T, H, W 在这里代表整个图像/视频的原始网格尺寸，
    #                     # 而不是前景或背景的独立尺寸
    #                     # 你需要从 image_grid_thw 或 video_grid_thw 中根据 current_image_original_grid_idx/current_video_original_grid_idx 获取
    #                     # 假设 for 循环外我们知道哪个是图片哪个是视频，或者通过 `vision_tokens` 判断
    #                     t, h, w = (
    #                         image_grid_thw[current_image_original_grid_idx] if image_token_id in vision_tokens else video_grid_thw[current_video_original_grid_idx]
    #                     )
    #                     llm_grid_t, llm_grid_h, llm_grid_w = (
    #                         t.item(),
    #                         h.item() // spatial_merge_size,
    #                         w.item() // spatial_merge_size,
    #                     )

    #                     # 生成所有逻辑 Patch 的原始 3D 位置，然后我们再从中选取背景和前景
    #                     # 这一步是 Qwen 原始逻辑中生成图像/视频 3D 位置 ID 的部分
    #                     t_indices_full = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
    #                     h_indices_full = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
    #                     w_indices_full = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        
    #                     # 假设你可以根据 is_background_mask_logical 再次生成一个排序
    #                     # 实际上，这里你需要的是 bg_original_logical_patch_ids_window_order 和 fg_original_logical_patch_ids_window_order
    #                     # 这些是 ViT 已经输出的，代表了按背景在前、前景在后重排后的原始逻辑 ID
    #                     # 为了简化，我假设你的 ViT 会返回 `bg_original_logical_patch_ids_window_order` 和 `fg_original_logical_patch_ids_window_order`
    #                     # 并在 Qwen2VLChat 的 forward 中将它们作为参数传递给 get_rope_index
    #                     # 假设这些 ids 是平坦的，分别对应 bg_features_final_ordered 和 fg_features_final_ordered
                        
    #                     # 假设你将 `original_logical_patch_ids_in_window_order` (bg + fg 拼接后) 传递过来
    #                     # 这个张量的形状是 [总逻辑 Patch 数量]
    #                     # 然后用它来索引 full_3d_indices
                        
    #                     # 临时：为了让这段代码不报错，我先用一个占位符。
    #                     # 这里的 `bg_original_logical_patch_ids_window_order_for_llm`
    #                     # 应该是从 CustomQwen2_5_VisionTransformerPretrainedModel 返回并传递过来的
    #                     # 并且它应该包含了背景和前景按照 ViT 输出顺序的原始逻辑 ID
                        
    #                     # 假设我们从某个地方获取到 `global_ordered_logical_patch_ids`
    #                     # 这个 ids 列表就是你在 ViT 中最终 `torch.cat` 时的顺序：[背景_原ID1, 背景_原ID2, ..., 前景_原ID1, 前景_原ID2, ...]
    #                     # 并且其长度是 `total_visual_logical_patches_current_item`
    #                     global_ordered_logical_patch_ids = self._get_global_ordered_logical_patch_ids_from_vit(i) # <-- 这是一个占位符函数，你需要实现它来获取

    #                     # 从完整的 3D 索引中根据新的顺序选择
    #                     full_3d_indices = torch.stack([t_indices_full, h_indices_full, w_indices_full]) # 形状 (3, total_logical_patches_current_item)
                        
    #                     # 根据新的顺序重新排列 3D 索引
    #                     reordered_3d_indices = full_3d_indices[:, global_ordered_logical_patch_ids]

    #                     # 将重排后的 3D 索引添加到 llm_pos_ids_list
    #                     llm_pos_ids_list.append(reordered_3d_indices + current_sequence_start_idx)
    #                     current_sequence_start_idx += total_visual_logical_patches_current_item

    #                 # 处理视觉token之后的文本部分
    #                 remaining_text_len = len(input_tokens_list) - current_sequence_start_idx
    #                 if remaining_text_len > 0:
    #                     st_idx = current_sequence_start_idx
    #                     llm_pos_ids_list.append(torch.arange(remaining_text_len).view(1, -1).expand(3, -1) + st_idx)
    #             else: # 纯文本情况
    #                 st_idx = 0
    #                 llm_pos_ids_list.append(torch.arange(len(input_tokens_list)).view(1, -1).expand(3, -1) + st_idx)


    #             llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    #             # 确保填充的部分位置 ID 不被修改
    #             position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
    #             mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids_batch)) # 注意这里使用 input_ids_batch

    #         mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    #         return position_ids, mrope_position_deltas
    #     else:
    #         # 原始纯文本逻辑
    #         if attention_mask is not None:
    #             position_ids = attention_mask.long().cumsum(-1) - 1
    #             position_ids.masked_fill_(attention_mask == 0, 1)
    #             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
    #             max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
    #             mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    #         else:
    #             position_ids = (
    #                 torch.arange(input_ids.shape[1], device=input_ids.device)
    #                 .view(1, 1, -1)
    #                 .expand(3, input_ids.shape[0], -1)
    #             )
    #             mrope_position_deltas = torch.zeros(
    #                 [input_ids.shape[0], 1],
    #                 device=input_ids.device,
    #                 dtype=input_ids.dtype,
    #             )
    #         return position_ids, mrope_position_deltas
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # 接收外部传入的标志和mask
        use_image_segmentation: bool = True, 
        is_foreground_mask: Optional[torch.Tensor] = None, # 直接接收已经处理好的 mask
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                
                image_embeds, final_is_foreground_mask = self.visual(
                    pixel_values=pixel_values,
                    grid_thw=image_grid_thw, # 图像的grid_thw
                    is_foreground_mask=is_foreground_mask # 传递给视觉模型
                )
                
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # *** 修改点 2: 在这里执行重排操作 ***
        # 这个操作通常只在需要缓存的推理阶段进行
        if use_image_segmentation:
            # 假设 batch_size = 1
            batch_size, seq_len, _ = inputs_embeds.shape
            if batch_size != 1:
                raise NotImplementedError("Reordering logic currently supports batch_size=1 only.")

            # --- 核心重排逻辑开始 ---
            
            # *** 步骤 2: 找到所有图像token在完整序列中的位置 ***
            image_token_indices = (input_ids[0] == self.config.image_token_id).nonzero(as_tuple=True)[0]
            
            # 如果没有图像token，则无需做任何事
            if len(image_token_indices) > 0:
                # *** 步骤 3: 根据ViT的掩码，计算图像块内部的新顺序 ***
                # final_is_foreground_mask 与 image_token_indices 是一一对应的
                background_patch_mask = ~final_is_foreground_mask
                
                # 从全局索引中分离出背景和前景部分
                bg_image_indices = image_token_indices[background_patch_mask]
                fg_image_indices = image_token_indices[final_is_foreground_mask]
                
                # 这就是图像token在重排后，应该对应到的【原始位置索引】
                new_image_order_indices = torch.cat([bg_image_indices, fg_image_indices])
                
                # *** 步骤 4: 构建最终的、可用于gather操作的完整重排索引 ***
                # 创建一个单位映射 (0->0, 1->1, ...)
                full_reorder_indices = torch.arange(seq_len, device=inputs_embeds.device)
                
                # 将图像块部分的映射，替换为我们计算好的新顺序
                # 含义是：在新的序列中，原来 image_token_indices[i] 的位置，现在要从 new_image_order_indices[i] 去拿数据
                full_reorder_indices[image_token_indices] = new_image_order_indices
                
                # *** 步骤 5: 应用重排 ***
                # 使用 full_reorder_indices 来 gather 数据，实现重排
                inputs_embeds = inputs_embeds[:, full_reorder_indices, :]
                position_ids = position_ids[..., full_reorder_indices]
                
                if attention_mask is not None:
                    # 构造一个用于重排 atención mask 的 gather index
                    gather_index_for_attn = full_reorder_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, seq_len)
                    # 对于 4D mask (B, 1, S, S), 在 S x S 矩阵上进行行列重排
                    if attention_mask.ndim == 4:
                        # 重排行
                        attention_mask = torch.gather(attention_mask, 2, gather_index_for_attn)
                        # 重排列
                        attention_mask = torch.gather(attention_mask, 3, gather_index_for_attn.transpose(2,3))
                    elif attention_mask.ndim == 2:
                        attention_mask = attention_mask[:, full_reorder_indices]

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
