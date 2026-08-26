"""
SAM 사전학습 가중치를 쓰지 않는(config.NO_SAM=True) 경우의 대체 이미지 인코더.

detectron2의 ViT(base) 구현을 백본으로 사용하고, MAE(ImageNet)로 사전학습된
체크포인트(`sam_ckpts/mae_pretrain_vit_base.pth`)를 불러와 초기화한다.
model.py의 SAMRoadplus.__init__에서 NO_SAM 설정일 때 SAM ImageEncoderViT 대신
이 VITBEncoder를 사용한다. 출력은 SAM 인코더와 동일한 [B, output_feature_dim, H/16, W/16]
형태가 되도록 1x1 conv(output_feature_proj)로 채널을 맞춘다.
"""

import sys

sys.path.insert(0, "./detectron2")

from functools import partial

import torch.nn as nn
import torch
import torch.nn.functional as F


from detectron2.modeling import ViT
import pprint


class VITBEncoder(nn.Module):
    def __init__(self, image_size, output_feature_dim):
        super(VITBEncoder, self).__init__()
        # Base
        embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
        # Creates Simple Feature Pyramid from ViT backbone
        self.vitb = ViT(  # Single-scale ViT backbone
            img_size=image_size,
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=dp,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[
                # 2, 5, 8 11 for global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
            ],
            residual_block_indexes=[],
            use_rel_pos=True,
            out_feature="last_feat",
        )

        self.output_feature_proj = nn.Conv2d(
            embed_dim, output_feature_dim, kernel_size=1, stride=1
        )

        # MAE(ImageNet) 사전학습 가중치를 불러와서, shape이 일치하는 파라미터만 골라 로드한다.
        # (SAM 체크포인트가 아니므로 클래스 헤드 등 구조가 다른 파라미터는 자연히 mismatch 처리됨)
        with open("sam_ckpts/mae_pretrain_vit_base.pth", "rb") as f:
            ckpt_state_dict = torch.load(f)["model"]
            ckpt_state_dict = {"vitb." + k: v for k, v in ckpt_state_dict.items()}

            # for k, v in ckpt_state_dict.items():
            #     print(f'ckpt {k} shape {v.shape}')

            ## Resize pos embeddings, if needed
            # if image_size != 1024:
            #     new_state_dict = resize_vit_pos_embed(ckpt_state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
            #     ckpt_state_dict = new_state_dict

            matched_names = []
            mismatch_names = []
            state_dict_to_load = {}
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]
                else:
                    mismatch_names.append(k)
            print("###### Matched params ######")
            pprint.pprint(matched_names)
            print("###### Mismatched params ######")
            pprint.pprint(mismatch_names)

            self.matched_param_names = set(matched_names)
            self.vitb.load_state_dict(state_dict_to_load, strict=False)

    def forward(self, x):
        x = self.vitb(x)["last_feat"]
        x = self.output_feature_proj(x)
        return x


if __name__ == "__main__":
    img_size = 512
    img = torch.zeros((1, 3, img_size, img_size), dtype=torch.float32)
    vitb = VITBEncoder(img_size, 256)
    x = vitb(img)
    print(x.shape)
