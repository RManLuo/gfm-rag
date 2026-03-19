import hashlib
import json
import logging
import os

import dotenv
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gfmrag import GraphIndexer

logger = logging.getLogger(__name__)

dotenv.load_dotenv()


def get_tmp_dir(cfg: DictConfig) -> str:
    """Get the temporary directory by the config"""

    # create a fingerprint of config for tmp directory
    config = OmegaConf.to_container(cfg, resolve=True)
    if "force" in config:
        del config["force"]
    if "force" in config.get("el_model", {}):
        del config["el_model"]["force"]
    fingerprint = hashlib.md5(json.dumps(config).encode()).hexdigest()

    base_tmp_dir = os.path.join(cfg.root, fingerprint)
    if not os.path.exists(base_tmp_dir):
        os.makedirs(base_tmp_dir)
        json.dump(
            config,
            open(os.path.join(base_tmp_dir, "config.json"), "w"),
            indent=4,
        )

    return base_tmp_dir


@hydra.main(
    config_path="config/gfm_rag", config_name="index_dataset", version_base=None
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    graph_tmp_dir = get_tmp_dir(cfg.graph_constructor)
    sft_tmp_dir = get_tmp_dir(cfg.sft_constructor)
    graph_constructor = instantiate(cfg.graph_constructor, root=graph_tmp_dir)
    sft_constructor = instantiate(cfg.sft_constructor, root=sft_tmp_dir)

    kg_indexer = GraphIndexer(graph_constructor, sft_constructor)
    kg_indexer.index_data(cfg.dataset)


if __name__ == "__main__":
    main()
