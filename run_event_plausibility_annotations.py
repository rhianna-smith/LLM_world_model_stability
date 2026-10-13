import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from modules.event_plausibility_annotations import run_event_likelihood_annotations

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="event_plausibility_annotations",
)
def main(cfg: DictConfig) -> None:
    llm_client = instantiate(cfg.model.client)
    run_event_likelihood_annotations(cfg, llm_client)


if __name__ == "__main__":
    main()
