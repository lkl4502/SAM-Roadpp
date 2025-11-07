import cv2
import torch
import matplotlib.pyplot as plt

from utils import load_config


def kuma_variance_map(path, ckpt_path, img_path, gt_path, output_name):
    from model_kuma import SAMRoadplus

    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    img = img[64:576, 64:576, :]
    gt = gt[64:576, 64:576]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    config = load_config(path)
    checkpoint = torch.load(ckpt_path)

    model = SAMRoadplus(config)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    kuma_a_b = model(img_tensor, None, None, None)[0][0]  # 512, 512, 2

    kuma_a_b = torch.exp(kuma_a_b) + 0.1

    a = kuma_a_b[..., 0]  # 512, 512
    b = kuma_a_b[..., 1]  # 512, 512

    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.imshow(a.detach().cpu(), cmap="viridis")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.title("a map")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.imshow(b.detach().cpu(), cmap="viridis")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.title("b map")
    # plt.axis("off")

    # plt.tight_layout()
    # plt.savefig(output_name)

    # Beta(z1, z2) = (gamma(z1) * gamma(z2)) / gamma(z1 + z2)
    log_beta_1 = (torch.lgamma(1 + 1 / a) + torch.lgamma(b)) - torch.lgamma(
        1 + 1 / a + b
    )
    log_beta_2 = (torch.lgamma(1 + 2 / a) + torch.lgamma(b)) - torch.lgamma(
        1 + 2 / a + b
    )

    beta_1 = torch.exp(log_beta_1)
    beta_2 = torch.exp(log_beta_2)

    # m_n = b * Beta(1 + n / a, b)
    m_1 = b * beta_1
    m_2 = b * beta_2

    # var = m_2 - m_1^2
    var = m_2 - m_1**2
    var = var.clamp(min=0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(var.detach().cpu(), cmap="magma")
    plt.title("Variance (Uncertainty) Map")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_name)


def variance_map_heatmap(config_path, ckpt_path, img_path, gt_path, output_name):
    from model_EMA import SAMRoadplus

    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    img = img[64:576, 64:576, :]
    gt = gt[64:576, 64:576]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    config = load_config(config_path)
    checkpoint = torch.load(ckpt_path)

    model = SAMRoadplus(config)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    mask_logits_list, _, mask_road_vars_list, mask_keypoint_vars_list = model(
        img_tensor, None, None, None
    )  # 512, 512, 2

    logits = mask_logits_list[0][0]
    keypoint_logits = logits[..., 0]
    road_logits = logits[..., 1]

    mask_road_var = mask_road_vars_list[0][0].squeeze(-1)
    mask_keypoint_var = mask_keypoint_vars_list[0][0].squeeze(-1)

    keypoint_var = torch.exp(mask_keypoint_var) + 0.1
    road_var = torch.exp(mask_road_var) + 0.1

    keypoint_denominator = torch.sqrt(1 + torch.pi**2 / 8 * keypoint_var)
    road_denominator = torch.sqrt(1 + torch.pi**2 / 8 * road_var)

    keypoint_probit = torch.sigmoid(keypoint_logits / keypoint_denominator) > 0.242
    road_probit = torch.sigmoid(road_logits / road_denominator) > 0.404

    road_var_map = torch.exp(mask_road_var)
    road_var_map = mask_road_var.clamp(max=2)
    keypoint_var_map = torch.exp(mask_keypoint_var)
    keypoint_var_map = keypoint_var_map.clamp(max=1)

    plt.figure(figsize=(18, 12))
    plt.subplot(2, 3, 1)
    plt.imshow(road_var_map.detach().cpu(), cmap="magma")
    plt.title("Road Variance Map")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(keypoint_var_map.detach().cpu(), cmap="magma")
    plt.title("Keypoint Variance Map")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(road_probit.detach().cpu(), cmap="gray")
    plt.title("Road Probit")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(keypoint_probit.detach().cpu(), cmap="gray")
    plt.title("Keypoint Probit")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(img)
    plt.title("RGB")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_name)


# EMA variance map
ckpt_path = "/home/lkl4502/SAM-Roadpp/lightning_logs/8coq27xv/checkpoints/epoch=27-step=4032.ckpt"
config_path = (
    "/home/lkl4502/SAM-Roadpp/config/city-scale/toponet_vitb_512/EMA_0.9999.yaml"
)


# kuma_variance_map
# path = "/home/lkl4502/SAM-Roadpp/config/city-scale/toponet_vitb_512/kuma_1e-1.yaml"
# ckpt_path = "/home/lkl4502/SAM-Roadpp/lightning_logs/kuma_1e-1_108/checkpoints/epoch=10-step=1584.ckpt"
# img_path = "/home/lkl4502/data/Aerial/RoadGraph/cityscale/20cities/region_108_sat.png"
# gt_path = "/home/lkl4502/data/Aerial/RoadGraph/cityscale/20cities/region_108_gt.png"
img_path = "/home/lkl4502/data/Global-Scale/Global-Scale/out_of_domain/region_0_sat.png"
gt_path = "/home/lkl4502/data/Global-Scale/Global-Scale/out_of_domain/region_0_gt.png"
# kuma_variance_map(path, ckpt_path, img_path, gt_path, "kuma_variance_map2.png")

variance_map_heatmap(config_path, ckpt_path, img_path, gt_path, "variance_map.png")
