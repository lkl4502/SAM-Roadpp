import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import math
import copy
from functools import partial
from torchmetrics.classification import BinaryJaccardIndex, F1Score, BinaryPrecisionRecallCurve
import lightning.pytorch as pl
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.common import LayerNorm2d
import numpy as np
import wandb
import pprint
import torchvision
import vitdet


class BilinearSampler(nn.Module):
    def __init__(self, config):
        super(BilinearSampler, self).__init__()
        self.config = config

    def forward(self, feature_maps, sample_points):
        """
        Args:
            feature_maps (Tensor): The input feature tensor of shape [B, D, H, W].
            sample_points (Tensor): The 2D sample points of shape [B, N_points, 2],
                                    each point in the range [-1, 1], format (x, y).
        Returns:
            Tensor: Sampled feature vectors of shape [B, N_points, D].
        """
        B, D, H, W = feature_maps.shape
        _, N_points, _ = sample_points.shape

        # normalize cooridinates to (-1, 1) for grid_sample
        sample_points = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0
        
        # sample_points from [B, N_points, 2] to [B, N_points, 1, 2] for grid_sample
        sample_points = sample_points.unsqueeze(2)
        # Use grid_sample for bilinear sampling. Align_corners set to False to use -1 to 1 grid space.
        # [B, D, N_points, 1]
        sampled_features = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)
        
        # sampled_features is [B, N_points, D]
        sampled_features = sampled_features.squeeze(dim=-1).permute(0, 2, 1)
        return sampled_features
     
def extract_point(x1,y1,x2,y2,image,num_points):
    H, W = image.shape[-2:] 
    x_values = torch.linspace(0, 1, steps=num_points).unsqueeze(0).unsqueeze(0).to(image.device)
    y_values = torch.linspace(0, 1, steps=num_points).unsqueeze(0).unsqueeze(0).to(image.device)#uniform sampling
    
    x_interp = x1.unsqueeze(-1) + (x2 - x1).unsqueeze(-1) * x_values
    y_interp = y1.unsqueeze(-1) + (y2 - y1).unsqueeze(-1) * y_values
    
    x_interp = torch.clamp(x_interp.long(), min=0, max=W-1)
    y_interp = torch.clamp(y_interp.long(), min=0, max=H-1)
    
    x_plus_1 = torch.clamp(x_interp + 1, max=W-1)
    y_plus_1 = torch.clamp(y_interp + 1, max=H-1)
   
    x_final = torch.cat([x_interp,x_interp,x_plus_1], dim=-1)
    y_final = torch.cat([y_interp,y_plus_1,y_interp], dim=-1)

    return(x_final,y_final)# sample points

def extendline(points1, points2, image):
    """Using linear interpolation to get pixels between two points in batch."""
    B, N, _ = points1.shape  # B: batch size, N: number of point pairs
    H, W = image.shape[-2:]
    height, width = H,W
    extend_length=8 #extend length

    batch_A = points1
    batch_B = points2
    # direction vector
    directions = batch_B - batch_A  # (N, 2)

    directions = directions.float() 
    lengths = torch.norm(directions, dim=2, keepdim=True)  # (N, 1)
    lengths = lengths.masked_fill(lengths == 0, 1e-8)
    directions_norm = directions / lengths  # (N, 2)

    extended_A = batch_A - directions_norm * extend_length  # (N, 2)
    extended_B = batch_B + directions_norm * extend_length  # (N, 2)
    # Round the coordinates to integers
    extended_A = torch.round(extended_A).long()
    extended_B = torch.round(extended_B).long()
    
    extended_A[:, 0] = extended_A[:, 0].clamp(0, width - 1)
    extended_A[:, 1] = extended_A[:, 1].clamp(0, height - 1)
    extended_B[:, 0] = extended_B[:, 0].clamp(0, width - 1)
    extended_B[:, 1] = extended_B[:, 1].clamp(0, height - 1)

    extend_x1,extend_y1 = extended_A[...,0],extended_A[...,1]
    extend_x2,extend_y2 = extended_B[...,0],extended_B[...,1]

    x1, y1 = points1[..., 0], points1[..., 1]
    x2, y2 = points2[..., 0], points2[..., 1]

    x_final_1,y_final_1  = extract_point(extend_x1,extend_y1,x1,y1,image,num_points =15 )#sample extend point one side
    x_final,y_final  = extract_point(x1,y1,x2,y2,image,num_points =20 ) #sample between point
    x_final_2,y_final_2  = extract_point(extend_x2,extend_y2,x2,y2,image,num_points=15 )#sample extend point other side

    features1 = image[np.arange(B)[:, None, None], x_final_1,  y_final_1]
    features = image[np.arange(B)[:, None, None], x_final,  y_final]
    features2 = image[np.arange(B)[:, None, None], x_final_2,  y_final_2]#extract mask feature
    features = torch.concat([features1,features,features2], dim=2)

    return features

class TopoNet(nn.Module):
    def __init__(self, config, feature_dim):
        super(TopoNet, self).__init__()
        self.config = config

        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3
        #self.hidden_dim2 = 256
        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)
        self.pair_proj = nn.Linear(2 * self.hidden_dim + 152, self.hidden_dim)

        # Create Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True  # Input format is [batch size, sequence length, features]
        )
        
        # Stack the Transformer Encoder Layers
        if self.config.TOPONET_VERSION != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features,points1, point_features1, pairs, pairs_valid,mask_scores):
        # points: [B, N_points, 2]
        # point_features: [B, N_points, D]
        # pairs: [B, N_samples, N_pairs, 2]
        # pairs_valid: [B, N_samples, N_pairs]
        # mask scores:[B,3,512,512]
        B,_,H,W = mask_scores.shape
        point_features = F.relu(self.feature_proj(point_features))
        # gathers pairs
        batch_size, n_samples, n_pairs, _ = pairs.shape
        pairs = pairs.view(batch_size, -1, 2)
        
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, n_samples * n_pairs)
        # Use advanced indexing to fetch the corresponding feature vectors
        # [B, N_samples * N_pairs, D]
        src_features = point_features[batch_indices, pairs[:, :, 0]]
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]
        # [B, N_samples * N_pairs, 2]
        src_points = points[batch_indices, pairs[:, :, 0]]
        tgt_points = points[batch_indices, pairs[:, :, 1]]

        _,N,_ = tgt_points.shape

        mask_road_dim = mask_scores[:, 1, :, :] 
        line_features = extendline(src_points, tgt_points, mask_road_dim)#][B,N,64]
        offset_x = tgt_points - src_points
        ## ablation study
        # [B, N_samples * N_pairs, 2D + 2]
        if self.config.TOPONET_VERSION == 'no_tgt_features':
            pair_features = torch.concat([src_features, torch.zeros_like(tgt_features), offset_x], dim=2)
        if self.config.TOPONET_VERSION == 'no_offset':
            pair_features = torch.concat([src_features, tgt_features, torch.zeros_like(offset_x)], dim=2)
        else:
            pair_features = torch.concat([src_features, tgt_features, line_features,offset_x], dim=2)
        # [B, N_samples * N_pairs, D]
        pair_features = F.relu(self.pair_proj(pair_features))
        # attn applies within each local graph sample
        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        # valid->not a padding
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)

        # [B * N_samples, 1]
        #### flips mask for all-invalid pairs to prevent NaN
        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1)
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)

        padding_mask = ~pairs_valid
        
        ## ablation study
        if self.config.TOPONET_VERSION != 'no_transformer':
            pair_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask)
        
        ## Seems like at inference time, the returned n_pairs heres might be less - it's the
        # max num of valid pairs across all samples in the batch
        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)

        # [B, N_samples, N_pairs, 1]
        logits = self.output_proj(pair_features)

        scores = torch.sigmoid(logits)

        return logits, scores

class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        # self.qkv = qkv
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        # qkv = self.qkv(x)  # B,N,N,3*org_C
        qkv = F.linear(x, self.weight, self.bias)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

class SAMRoadplus(pl.LightningModule):
    """This is the RelationFormer module that performs object detection"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.SAM_VERSION in {'vit_b', 'vit_l', 'vit_h'}
        if config.SAM_VERSION == 'vit_b':
            ### SAM config (B)
            encoder_embed_dim=768
            encoder_depth=12
            encoder_num_heads=12
            encoder_global_attn_indexes=[2, 5, 8, 11]
            ###
        elif config.SAM_VERSION == 'vit_l':
            ### SAM config (L)
            encoder_embed_dim=1024
            encoder_depth=24
            encoder_num_heads=16
            encoder_global_attn_indexes=[5, 11, 17, 23]
            ###
        elif config.SAM_VERSION == 'vit_h':
            ### SAM config (H)
            encoder_embed_dim=1280
            encoder_depth=32
            encoder_num_heads=16
            encoder_global_attn_indexes=[7, 15, 23, 31]
            ###
            
        prompt_embed_dim = 256
        # SAM default is 1024
        image_size = config.PATCH_SIZE
        self.image_size = image_size
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        encoder_output_dim = prompt_embed_dim

        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        if self.config.NO_SAM:
            ### im1k + mae pre-trained vitb
            self.image_encoder = vitdet.VITBEncoder(image_size=image_size, output_feature_dim=prompt_embed_dim)
            self.matched_param_names = self.image_encoder.matched_param_names
        else:
            ### SAM vitb
            self.image_encoder = ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim
            )
        if self.config.USE_SAM_DECODER:
            # SAM DECODER
            # Not used, just produce null embeddings
            self.prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            )
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            self.mask_decoder=MaskDecoder(
                num_multimask_outputs=2, # keypoint, road
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=2,
                iou_head_hidden_dim=256,
            )
        else:
            #### Naive decoder
            activation = nn.GELU
            self.map_decoder = nn.Sequential(
                nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                activation(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
            )
        #### TOPONet
        self.bilinear_sampler = BilinearSampler(config)
        self.topo_net = TopoNet(config, 256)
        #### LORA
        if config.ENCODER_LORA:
            r = self.config.LORA_RANK
            lora_layer_selection = None
            assert r > 0
            if lora_layer_selection:
                self.lora_layer_selection = lora_layer_selection
            else:
                self.lora_layer_selection = list(
                    range(len(self.image_encoder.blocks)))  # Only apply lora to the image encoder by default
            # create for storage, then we can init them or load weights
            self.w_As = []  # These are linear layers
            self.w_Bs = []

            # lets freeze first
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            # Here, we do the surgery
            for t_layer_i, blk in enumerate(self.image_encoder.blocks):
                # If we only want few lora layer instead of all
                if t_layer_i not in self.lora_layer_selection:
                    continue
                w_qkv_linear = blk.attn.qkv
                dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            # Init LoRA params
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)
        #### Losses
        if self.config.FOCAL_LOSS:
            self.mask_criterion = partial(torchvision.ops.sigmoid_focal_loss, reduction='mean')
        else:
            self.mask_criterion = torch.nn.BCEWithLogitsLoss()
        self.topo_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        #### Metrics
        self.keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.road_iou = BinaryJaccardIndex(threshold=0.5)
        self.topo_f1 = F1Score(task='binary', threshold=0.5, ignore_index=-1)
        # testing only, not used in training
        self.keypoint_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.road_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.topo_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)

        if self.config.NO_SAM:
            return
        with open(config.SAM_CKPT_PATH, "rb") as f:
            ckpt_state_dict = torch.load(f)

            ## Resize pos embeddings, if needed
            if image_size != 1024:
                new_state_dict = self.resize_sam_pos_embed(ckpt_state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
                ckpt_state_dict = new_state_dict
            
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
            self.load_state_dict(state_dict_to_load, strict=False)

    def resize_sam_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        new_state_dict = {k : v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            # Copied from SAMed
            # resize pos embedding, which may sacrifice the performance, but I have no better idea
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        return new_state_dict
    
    def forward(self, rgb, graph_points, pairs, valid):
        # rgb: [B, H, W, C]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]
        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        # [B, D, h, w]
        image_embeddings = self.image_encoder(x)
        # mask_logits, mask_scores: [B, 2, H, W]
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(image_embeddings)
            mask_scores = torch.sigmoid(mask_logits)
        
        ## Predicts local topology
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        # [B, N_sample, N_pair, 1]
        topo_logits, topo_scores = self.topo_net(graph_points, point_features, pairs, valid,mask_scores)
        # [B, H, W, 2]
        mask_logits = mask_logits.permute(0, 2, 3, 1)
        mask_scores = mask_scores.permute(0, 2, 3, 1)
        return mask_logits, mask_scores, topo_logits, topo_scores
    
    def infer_masks_and_img_features(self, rgb):
        # rgb: [B, H, W, C]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]
        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        # [B, D, h, w]
        image_embeddings = self.image_encoder(x)
        # mask_logits, mask_scores: [B, 2, H, W]
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(image_embeddings)
            mask_scores = torch.sigmoid(mask_logits)
        
        # [B, H, W, 2]
        mask_scores = mask_scores.permute(0, 2, 3, 1)
        return mask_scores, image_embeddings
    
    def infer_toponet(self, image_embeddings, graph_points, pairs, valid,mask_scores):
        # image_embeddings: [B, D, h, w]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]
        ## Predicts local topology
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        # [B, N_sample, N_pair, 1]
        topo_logits, topo_scores = self.topo_net(graph_points, point_features, graph_points, point_features, pairs, valid,mask_scores)
        return topo_scores
