"""
    subclass for PyTorch Llava Model, add method to extra image and text representation
    after visual and textual encoder in mllms from the text-to-image datasets
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch, transformers
from transformers import (LlavaForConditionalGeneration,
                          LlavaConfig,
                          AutoModel,
                          AutoModelForCausalLM,
                          )
import os

@dataclass
class ImageAndTextEmbeddings:
    image_embeds: torch.FloatTensor
    text_embeds: torch.FloatTensor

class IdCor_LlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: LlavaConfig):
        super().__init__(config)

    def extra_imageAndtext_embeddings(
        self,
        human_input: List[str],
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

        # check the data in the batch
        batch_size = input_ids.shape[0]
        device = input_ids.device        

        # extra the text embeddings representating the total text information by fusion inputs_embeds using the attention
        if input_ids is not None and attention_mask is not None:
            # 使用父类method获取inputs_embeds
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # 作用mask，忽略padding_token的embeddings
            attention_mask_expanded = attention_mask.unsqueeze(-1)
            masked_input_ids = input_ids * attention_mask_expanded

        
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
