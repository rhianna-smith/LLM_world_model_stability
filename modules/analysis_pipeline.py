import os
from omegaconf import DictConfig, OmegaConf
from modules.analysis_data import ensure_analysis_output_dirs, filter_analysis_sample, load_analysis_data
from modules.analysis_margins import build_marginal_predictions, print_marginal_predictions, write_marginal_predictions
from modules.analysis_models import fit_configured_models
from modules.analysis_tables import models_to_table, print_model_outputs, write_model_outputs
from modules.console import print_banner, print_subbanner

def run_analysis(cfg: DictConfig) -> None:
    print_banner("Starting Regression Analysis")
    ensure_analysis_output_dirs(cfg)
    save_resolved_config(cfg)
    print_subbanner("Loading And Preparing Analysis Data")
    analysis_df = load_analysis_data(cfg)
    analysis_df.to_csv(
        os.path.join(cfg.paths.analysis_output_dir, "analysis_dataset.csv"),
        index=False,
    )
    print("Analysis rows: %s" % len(analysis_df))

    models = {}
    sample_dfs = {}
    for sample_cfg in cfg.analysis.samples:
        print_subbanner("Preparing Sample: %s" % sample_cfg.name)
        sample_df = filter_analysis_sample(
            analysis_df,
            sample_cfg,
            cfg.analysis.selected_rating_col,
        )
        sample_dfs[sample_cfg.name] = sample_df
        sample_df.to_csv(
            os.path.join(
                cfg.paths.analysis_output_dir,
                "%s_analysis_dataset.csv" % sample_cfg.name,
            ),
            index=False,
        )
        print("Sample rows: %s" % len(sample_df))
        print_subbanner("Fitting Models: %s" % sample_cfg.name)
        sample_models = fit_configured_models(sample_df, cfg)
        for model_name, model in sample_models.items():
            models["%s__%s" % (sample_cfg.name, model_name)] = model

    print_banner("Regression Tables")
    table_df = models_to_table(models)
    print_model_outputs(models, table_df)
    write_model_outputs(models, table_df, cfg.paths.analysis_output_dir)
    marginal_df = build_marginal_predictions(models, sample_dfs, cfg)
    print_marginal_predictions(marginal_df)
    write_marginal_predictions(marginal_df, cfg.paths.analysis_output_dir)
    print_banner("Analysis Artifacts Written")
    print("Output directory: %s" % cfg.paths.analysis_output_dir)


def save_resolved_config(cfg: DictConfig) -> None:
    config_path = os.path.join(cfg.paths.analysis_output_dir, "analysis_resolved_config.yaml")
    with open(config_path, "w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(cfg, resolve=True))
