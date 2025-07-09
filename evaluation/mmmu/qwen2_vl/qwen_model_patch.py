import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionBlock, 
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLVisionConfig,
    Qwen2_5_VLVisionAbstractModel,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLConfig,
    # 导入 Qwen 原始 Block 内部可能用到的辅助函数或类
    # 例如：RotaryEmbedding, get_window_index, PatchEmbed 等
    # 这里我们假设它们可以直接从 Qwen2_5_VLVisionTransformerPretrainedModel 实例中访问
    # 或者需要从 modeling_qwen2_5_vl.py 中导入
    # 为了简化，我们假设这些内部组件是可访问的
)
# 导入 Qwen 原始的 rotary_embedding 和 window_attention 相关的辅助函数
# 假设这些函数在 modeling_qwen2_5_vl.py 中或者可以从 self.visual 访问
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import TwoDContinuousRotaryEmbedding, get_window_index_func

# --- 自定义 Qwen2_5_VLVisionBlock ---
class CustomQwen2_5VLVisionBlock(Qwen2_5_VLVisionBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        window_index: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        # 新增：patch_attention_mask
        patch_attention_mask: Optional[torch.BoolTensor] = None, 
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        
        # 复制 Qwen2_5_VLVisionBlock 的原始 forward 逻辑
        # 假设原始 Block 的结构如下：
        # hidden_states -> norm1 -> attn -> add -> norm2 -> mlp -> add

        # Pre-attention normalization
        norm_hidden_states = self.norm1(hidden_states)

        # Self-attention
        # attention_output 应该是 (batch_size, seq_len, hidden_size)
        # attention_weights 应该是 (batch_size, num_heads, seq_len, seq_len)
        attention_output, attention_weights = self.attn(
            hidden_states=norm_hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            window_index=window_index,
            cu_window_seqlens=cu_window_seqlens,
            # 关键修改：传递 patch_attention_mask 给 Attention 模块
            # 假设 self.attn 是一个能够处理 attention_mask 的 MultiheadAttention 模块
            attention_mask=patch_attention_mask.unsqueeze(-1).unsqueeze(-1) if patch_attention_mask is not None else None, # (1, 1, seq_len, 1) or (1, 1, seq_len, seq_len)
            output_attentions=output_attentions,
        )

        # Residual connection
        hidden_states = hidden_states + attention_output

        # Pre-MLP normalization
        norm_hidden_states = self.norm2(hidden_states)

        # MLP
        mlp_output = self.mlp(norm_hidden_states)

        # Residual connection
        hidden_states = hidden_states + mlp_output

        if output_attentions:
            return hidden_states, attention_weights
        return hidden_states, None # 返回 None for attention_weights if not requested

# --- 自定义 Qwen2_5_VisionTransformerPretrainedModel ---
class CustomQwen2_5VisionTransformer(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config: Qwen2_5_VLVisionConfig):
        super().__init__(config)
        # 替换原始的 Block 列表为我们自定义的 Block
        self.blocks = nn.ModuleList([CustomQwen2_5VLVisionBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        # 新增：patch_attention_mask
        patch_attention_mask: Optional[torch.BoolTensor] = None, 
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Patch Embedding
        # pixel_values shape: (batch_size, channels, height, width)
        # hidden_states shape: (num_patches, hidden_size) for single image, or (batch_size * num_patches, hidden_size)
        hidden_states = self.patch_embed(pixel_values)

        # 2. Positional Embedding (2D-ROPE)
        # rotary_pos_emb shape: (num_patches, head_dim)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # 3. Window Index for Window Attention
        # window_index shape: (num_patches, num_windows_per_patch)
        # cu_window_seqlens shape: (num_windows + 1)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=pixel_values.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 4. Transformer Blocks
        for i, block_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Pass patch_attention_mask to the custom block
            layer_outputs = block_module(
                hidden_states,
                rotary_pos_emb,
                window_index,
                cu_window_seqlens,
                patch_attention_mask=patch_attention_mask, # 传递 mask
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 5. Final Normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states # 返回最终的图像嵌入

# 导入 Qwen 原始的 Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as OriginalQwen2_5_VLForConditionalGeneration

# --- 继承并修改 Qwen2_5_VLForConditionalGeneration ---
class CustomQwen2_5_VLForConditionalGeneration(OriginalQwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # 替换原始的 self.visual 为我们自定义的 CustomQwen2_5VisionTransformer
        self.visual = CustomQwen2_5VisionTransformer._from_config(config.vision_config)
        # 确保其他部分保持不变
        self.model = OriginalQwen2_5_VLForConditionalGeneration.model_class(config) # 使用原始的 Qwen2_5_VLModel
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None

        self.post_init()

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
        # 新增：传递 patch_attention_mask
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        # 仅用于信息传递，不直接用于计算
        background_patch_indices: Optional[torch.LongTensor] = None,
        foreground_patch_indices: Optional[torch.LongTensor] = None,
        total_patches_in_image: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                
                # 关键修改：将 patch_attention_mask 传递给 self.visual
                image_embeds = self.visual(
                    pixel_values, 
                    grid_thw=image_grid_thw, 
                    patch_attention_mask=patch_attention_mask # 传递 mask
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
                
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype))
                
            if pixel_values_videos is not None:
                # 视频处理逻辑
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

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw, # 仍然是整个图像的 grid_thw
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

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
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        # 新增：传递 patch_attention_mask
        patch_attention_mask=None,
        # 仅用于信息传递
        background_patch_indices=None,
        foreground_patch_indices=None,
        total_patches_in_image=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        # 仅在生成的第一步传递图像像素值和索引
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["patch_attention_mask"] = None # 在后续步骤中不需要传递 mask
            model_inputs["background_patch_indices"] = None
            model_inputs["foreground_patch_indices"] = None
            model_inputs["total_patches_in_image"] = None

        return model_inputs