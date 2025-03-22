"""
    subclass for PyTorch Transformers Llava Model, add method to extra image and text representation
    after visual and textual encoder in mllms from the text-image multi-modalities datasets
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch, transformers
from transformers import (LlavaForConditionalGeneration,
                          LlavaConfig,
                          AutoModel,
                          AutoModelForCausalLM,
                          AutoProcessor
                          )
import os
import torch

@dataclass
class ImageAndTextEmbeddings:
    image_embeds: torch.FloatTensor
    text_embeds: torch.FloatTensor

class IdCor_nobert_LlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: LlavaConfig):
        super().__init__(config)

    def extra_imageAndtext_embeddings(
        self,
        processor: AutoProcessor,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None
    ) -> ImageAndTextEmbeddings:
        """
        Extract text embeddings by bert model and image embeddings after the projection by the trainable multi_modal_projector
        Args:
            input_ids: torch.LongTensor with shape (batch_size, sequence_length)
                The input ids of the text.
            pixel_values: torch.FloatTensor with shape (batch_size, channels, height, width)
                The pixel values of the image.
            attention_mask: Optional[torch.Tensor] with shape (batch_size, sequence_length)
                The attention mask of the text.
        Returns:
            ImageAndTextEmbeddings: A dataclass contains image_embeds and text_embeds.
        """

        def SlidingWindowGetImStart(input_ids: torch.LongTensor, image_token_index: int) -> List[List[int]]:
            """
                获取input_ids中每行中的<image> token的起始位置
            """
            input_id_list = [input_id for input_id in torch.unbind(input_ids, dim=0)]
            im_start_indices_list: List[List[int]] = []
            for batch_index, input_id in enumerate(input_id_list):  # 遍历每行的input_id
                row_im_start_list = []
                row_im_mask = (input_id == image_token_index)
                row_im_start = None
                for mask_index, im_mask in enumerate(row_im_mask): # 获取每段图像token的起始位置
                    if im_mask == 1 and row_im_start is None:
                        row_im_start = mask_index
                    elif (im_mask == 0 or mask_index == input_id.shape[0] - 1) and (row_im_start is not None):
                        row_im_start_list.append(row_im_start)
                        row_im_start = None
                im_start_indices_list.append(row_im_start_list)
            
            return im_start_indices_list
                
        # check the data in the batch
        batch_size = input_ids.shape[0]
        device = input_ids.device        

        # extra the text embeddings representating the total text information by fusion inputs_embeds using the attention
        if input_ids is not None and attention_mask is not None:
            """
                TODO: 设计纯粹基于llava文本模型的文本特征提取和融合算法
            """
            # 此时的input_ids经过了tokenizer处理，呈现为以batch为组织，但包含了大量的<image> token占位符，需要过滤剩一个<image> token
            im_start_indices_list:List[List[int]] = SlidingWindowGetImStart(input_ids=input_ids, image_token_index=self.config.image_token_index) # 获取每行中的<image> token区间的起始位置
            text_mask = (input_ids != self.config.image_token_index)   # 构造掩码标识出text token的位置, 每行input_ids的<image> token的个数一致
            
            for batch_index, row_im_start_indices in enumerate(im_start_indices_list):
                for im_start_indice in row_im_start_indices:
                    text_mask[batch_index][im_start_indice] = 1 # 每行的input_ids的<image> token区间段保留第一个<image> token
            
            text_input_ids_list = [row_input_ids[row_mask].unsqueeze(0) for row_input_ids, row_mask in zip(input_ids, text_mask)] # 获取每行input_ids中的文本部分, 组织成列表形式
            
            # 列表text_input_ids中的每个元素可能长度不一致，需要padding到相同长度对齐
            text_input_len_list = [row_input_ids.shape[1] for row_input_ids in text_input_ids_list] # row_input_ids: torch.size([1, ...])
            max_input_len = max(text_input_len_list)  # 计算所有行的最长token长度
            
            final_text_input_ids = torch.concat([   # padding到相同长度, 重新组织为(batch_size, max_input_len)的张量
                torch.concat([
                    torch.full(size=(1, max_input_len - text_input_len_list[index]), fill_value=processor.tokenizer.pad_token_id, device=input_ids.device),
                    value
                ], axis=1)
                for index, value in enumerate(iterable=text_input_ids_list)
            ])
            
            # 更新attention_mask
            final_attention_mask = torch.ones_like(final_text_input_ids)
            final_attention_mask[final_text_input_ids == processor.tokenizer.pad_token_id] = 0

            # 使用父类method获取inputs_embeds
            inputs_embeds = self.get_input_embeddings()(final_text_input_ids)

            # 应用自注意力机制进行原先所有的文本特征的融合
            nopad_final_text_embeds_list = [input_embed[mask.bool()] for input_embed, mask in zip(inputs_embeds, final_attention_mask)] # 在每个batch_index下筛选出非pad_token_id的embeddings, 组织成列表形式
            final_text_mean_embeds = [nopad_final_text_embeds.mean(dim=0) if nopad_final_text_embeds.numel() > 0 else torch.zeros(nopad_final_text_embeds.shape[1]) for nopad_final_text_embeds in nopad_final_text_embeds_list]    # 计算每个batch的平均embeddings
            final_text_mean_embeds = torch.stack(final_text_mean_embeds, dim=0) # 组织成(batch_size, embeddings)形状的张量
            
            final_text_mean_embeds.to(device)
            text_embeds = torch.empty(0, final_text_mean_embeds.shape[-1], device=device)

            # attention compute!
            for index in range(final_text_mean_embeds.shape[0]):
                Q_mean_text_embed = final_text_mean_embeds[index].unsqueeze(0)
                K_embeds = nopad_final_text_embeds_list[index]
                V_embeds = nopad_final_text_embeds_list[index]

                W_embeds = torch.matmul(Q_mean_text_embed, K_embeds.T)
                W_embeds = torch.nn.functional.softmax(W_embeds, dim=-1)
                final_embeds = torch.matmul(W_embeds, V_embeds)
                text_embeds = torch.cat((text_embeds, final_embeds), dim=0)

        # get the origin image_features after the projection and use the self-attention mechanism to get the total image embeddings
        if pixel_values is not None:
            # 使用父类method获取image_features
            image_features = self.get_image_features(pixel_values=pixel_values, 
                                                    vision_feature_layer=vision_feature_layer, 
                                                    vision_feature_select_strategy=vision_feature_select_strategy
                                                    )
            
            # 计算每个图像的初始平均feature, image_features_average.shape:(batch_size, config.text_config.hidden_size)
            image_features_average = torch.mean(image_features, dim=1)
            
            # 应用自注意力机制进行原先所有的图像特征的融合
            image_features.to(device), image_features_average.to(device)
            image_embeds = torch.empty(0, image_features_average.shape[-1], device=device)

            # attention compute!
            for index in range(image_features_average.shape[0]):    # 遍历一个batch中的数据
                Q_average_feature = image_features_average[index].unsqueeze(0)
                K_feature = image_features[index]
                V_feature = image_features[index]
                
                # divide the sqrt of the K_feature.shape[-1]
                W_feature = torch.matmul(Q_average_feature, K_feature.T)/torch.tensor(K_feature.shape[-1], dtype=torch.float32)     # divide
                W_feature = torch.nn.functional.softmax(W_feature, dim=-1)  # softmax
                final_feature = torch.matmul(W_feature, V_feature)
                image_embeds = torch.cat((image_embeds, final_feature), dim=0)

        # 返回ImageAndTextEmbeddings
        return ImageAndTextEmbeddings(
            image_embeds=image_embeds,
            text_embeds=text_embeds
        )
