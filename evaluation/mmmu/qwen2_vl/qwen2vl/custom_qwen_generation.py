import torch

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast, Qwen2VLForConditionalGeneration, Qwen2VLVisionConfig

# 确保导入所有 Qwen2.5-VL 原始模型依赖，如 Qwen2_5_VLModel, Qwen2_5_VLCausalLMOutputWithPast 等
from typing import Optional, List, Union, Tuple

# 从第一步导入我们自定义的 Vision Transformer 模型
from .custom_qwen_vision_transformer import CustomQwen2_VisionTransformerPretrainedModel 


from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from torch.nn import CrossEntropyLoss


_CONFIG_FOR_DOC = "Qwen2VLConfig"


class CustomQwen2_VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    def __init__(self, config: Qwen2VLVisionConfig):
        super().__init__(config)
        self.visual = CustomQwen2_VisionTransformerPretrainedModel._from_config(config.vision_config)

        
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
        # 接收外部传入的标志和mask
        use_image_segmentation: bool = True, 
        is_foreground_mask: Optional[torch.Tensor] = None, # 直接接收已经处理好的 mask
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())

                image_embeds, vit_reorder_indices = self.visual(
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
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
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
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)


        # === 修改点 2: 在送入模型前，根据ViT的重排索引，同步重排 position_ids 和 attention_mask ===
        if use_image_segmentation and pixel_values is not None:
            # 假设 batch_size = 1
            batch_size, seq_len, _ = inputs_embeds.shape
            if batch_size != 1:
                raise NotImplementedError("Reordering logic for BG/FG separation supports batch_size=1 only.")

            # a. 找到所有图像token在完整序列中的原始位置
            original_image_indices = (input_ids[0] == self.config.image_token_id).nonzero(as_tuple=True)[0]

            if len(original_image_indices) > 0:
                # b. vit_reorder_indices 是ViT内部对patch的重排，我们用它来重排图像token对应的全局位置
                # 例如，原始图像位置是[10,11,12], vit重排是[2,0,1], 那么新顺序就是[12,10,11]
                shuffled_global_indices = original_image_indices[vit_reorder_indices]

                # c. 创建一个完整的序列重排索引 (gather index)
                # 默认情况下，所有token位置不变 (0->0, 1->1, ...)
                full_reorder_indices = torch.arange(seq_len, device=inputs_embeds.device)
                # 对于图像token的位置，我们用新的顺序替换
                # 这意味着，新序列中原本是第10个token的位置，现在要去拿原始序列中第12个token的内容
                full_reorder_indices[original_image_indices] = shuffled_global_indices

                # d. 应用这个完整的重排索引到所有相关输入上
                # 注意：inputs_embeds 已经通过 masked_scatter 拥有了重排后的内容，所以我们【不需要】对它重排。
                # 我们只需要重排 position_ids 和 attention_mask 来与它对齐。
                position_ids = position_ids[..., full_reorder_indices]

                if attention_mask is not None:
                    if attention_mask.ndim == 4: # 形状 (B, 1, S, S)
                        attention_mask = attention_mask[:, :, full_reorder_indices, :]
                        attention_mask = attention_mask[:, :, :, full_reorder_indices]
                    elif attention_mask.ndim == 2: # 形状 (B, S)
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

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )