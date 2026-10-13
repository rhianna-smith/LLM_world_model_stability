import hydra
import logging
import litellm
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
from hydra.utils import instantiate
from omegaconf import DictConfig
from modules.pipeline import run_experiment

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    llm_client = instantiate(cfg.model.client)
    run_experiment(cfg, llm_client)


if __name__ == "__main__":
    main()
