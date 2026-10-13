import os
import pandas
from omegaconf import DictConfig


def load_analysis_data(cfg: DictConfig) -> pandas.DataFrame:
    analysis_df = pandas.read_csv(cfg.paths.analysis_dataset_path)

    type_df = pandas.read_csv(cfg.paths.type_columns_path)

    analysis_df = analysis_df.merge(
        type_df[
            [
                cfg.analysis.merge_key,
                "detail_type",
                "option_pair_type",
            ]
        ],
        on=cfg.analysis.merge_key,
        how="left",
    )

    if analysis_df["detail_type"].isna().any():
        raise ValueError("Some rows are missing detail_type after merge.")

    if analysis_df["option_pair_type"].isna().any():
        raise ValueError("Some rows are missing option_pair_type after merge.")

    return analysis_df

def ensure_analysis_output_dirs(cfg: DictConfig) -> None:
    os.makedirs(cfg.paths.analysis_output_dir, exist_ok=True)


def filter_analysis_sample(
    df: pandas.DataFrame,
    sample_cfg: object,
    selected_rating_col: str,
) -> pandas.DataFrame:
    sample_df = df.copy()
    if sample_cfg.min_selected_rating is not None:
        sample_df = sample_df[
            sample_df[selected_rating_col] >= sample_cfg.min_selected_rating
        ].copy()
    return sample_df



