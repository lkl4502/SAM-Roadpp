"""공통 유틸리티: config 로딩, wandb 시각화용 heatmap 변환, 출력 디렉터리 생성."""

import os
import io
import yaml
import wandb
import matplotlib.pyplot as plt

from PIL import Image
from addict import Dict
from datetime import datetime


def load_config(path):
    """YAML config 파일을 읽어 addict.Dict(속성처럼 . 접근 가능한 dict)로 반환한다."""
    with open(path) as file:
        config_dict = yaml.safe_load(file)
    return Dict(config_dict)


def tensor_to_heatmap(tensor):
    """2D 텐서를 magma 컬러맵 히트맵 이미지로 렌더링해 wandb.Image로 반환한다 (로깅용)."""
    fig, ax = plt.subplots()
    im = ax.imshow(tensor.cpu().numpy(), cmap="magma")
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)

    return wandb.Image(Image.open(buf))


def create_output_dir_and_save_config(output_dir_prefix, config, specified_dir=None):
    """추론 결과 저장용 디렉터리를 만들고(없으면 timestamp 기반 이름 생성), 사용된 config를
    재현 가능하도록 그 안에 config.yaml로 함께 저장한다."""
    if specified_dir:
        output_dir = specified_dir
    else:
        # Generate the output directory name with the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir_prefix}_{timestamp}"

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the path for the config file
    config_path = os.path.join(output_dir, "config.yaml")

    # Save the config as a YAML file
    with open(config_path, "w") as file:
        yaml.dump(config.to_dict(), file)

    return output_dir
