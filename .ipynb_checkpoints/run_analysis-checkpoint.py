import hydra
from omegaconf import DictConfig
from modules.analysis_pipeline import run_analysis

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="analysis",
)
def main(cfg: DictConfig) -> None:
    run_analysis(cfg)


if __name__ == "__main__":
    main()
