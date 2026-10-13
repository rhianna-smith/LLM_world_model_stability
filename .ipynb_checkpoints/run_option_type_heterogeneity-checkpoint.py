import hydra
from omegaconf import DictConfig
from rhianna_thesis.analysis_pipeline_heterogeneity import run_analysis


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="analysis_option_type_heterogeneity",
)
def main(cfg: DictConfig) -> None:
    run_analysis(cfg)


if __name__ == "__main__":
    main()
