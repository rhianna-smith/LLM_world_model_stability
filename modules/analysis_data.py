import os
import pandas
from omegaconf import DictConfig
from rhianna_thesis.console import print_subbanner

def load_analysis_data(cfg: DictConfig) -> pandas.DataFrame:
    results_df = load_judgment_results(cfg)
    ratings_df = pandas.read_csv(cfg.paths.ratings_path)
    analysis_df = results_df.merge(
        ratings_df,
        on=cfg.analysis.merge_key,
        how="inner",
        suffixes=(
            "",
            "_rating",
        ),
    )
    analysis_df = recode_analysis_data(analysis_df, cfg)
    return analysis_df


def load_judgment_results(cfg: DictConfig) -> pandas.DataFrame:
    result_dfs = []
    loaded_results = []
    skipped_paths = []
    for result_cfg in cfg.judgment_results:
        if not os.path.exists(result_cfg.path):
            skipped_paths.append(result_cfg.path)
            continue
        result_df = pandas.read_csv(result_cfg.path)
        result_df[cfg.analysis.judgment_model_col] = result_cfg.model
        result_dfs.append(result_df)
        loaded_results.append(
            "%s: %s rows from %s"
            % (
                result_cfg.model,
                len(result_df),
                result_cfg.path,
            )
        )

    if skipped_paths:
        print_subbanner("Skipped Missing Judgment Result Files")
        for path in skipped_paths:
            print(path)

    if not result_dfs:
        raise FileNotFoundError("No configured judgment result files were found.")

    print_subbanner("Loaded Judgment Result Files")
    for result in loaded_results:
        print(result)

    return pandas.concat(result_dfs, ignore_index=True)


def recode_analysis_data(df: pandas.DataFrame, cfg: DictConfig) -> pandas.DataFrame:
    df = df.copy()
    df[cfg.analysis.selected_rating_col] = df.apply(
        lambda row: select_chosen_likelihood_rating(
            row,
            cfg.analysis.forward_selection_col,
            cfg.analysis.option_1_rating_col,
            cfg.analysis.option_2_rating_col,
        ),
        axis=1,
    )
    df[cfg.analysis.selected_rating_source_col] = df.apply(
        lambda row: select_chosen_likelihood_source(
            row,
            cfg.analysis.forward_selection_col,
            cfg.analysis.option_1_rating_col,
            cfg.analysis.option_2_rating_col,
        ),
        axis=1,
    )

    numeric_cols = (
        list(cfg.analysis.predictor_cols)
        + list(cfg.analysis.lpm_outcomes)
        + list(cfg.analysis.logit_outcomes)
        + list(cfg.analysis.linear_outcomes)
    )
    for col in numeric_cols:
        df[col] = pandas.to_numeric(df[col])

    required_cols = (
        [
            cfg.analysis.merge_key,
            cfg.analysis.cluster_col,
            cfg.analysis.judgment_model_col,
            cfg.analysis.forward_selection_col,
            cfg.analysis.selected_rating_source_col,
        ]
        + list(cfg.analysis.predictor_cols)
        + list(cfg.analysis.lpm_outcomes)
        + list(cfg.analysis.logit_outcomes)
        + list(cfg.analysis.linear_outcomes)
    )
    analysis_df = df[unique_columns(required_cols)].dropna().copy()
    return analysis_df


def ensure_analysis_output_dirs(cfg: DictConfig) -> None:
    os.makedirs(cfg.paths.analysis_output_dir, exist_ok=True)


def filter_analysis_sample(
    df: pandas.DataFrame,
    sample_cfg: object,
    selected_rating_col: str
) -> pandas.DataFrame:
    sample_df = df.copy()
    if sample_cfg.min_selected_rating is not None:
        sample_df = sample_df[
            sample_df[selected_rating_col] >= sample_cfg.min_selected_rating
        ].copy()
    return sample_df


def select_chosen_likelihood_rating(
    row: pandas.Series,
    forward_selection_col: str,
    option_1_rating_col: str,
    option_2_rating_col: str
) -> object:
    if row[forward_selection_col] == "option_1":
        return row[option_1_rating_col]

    if row[forward_selection_col] == "option_2":
        return row[option_2_rating_col]

    return None


def select_chosen_likelihood_source(
    row: pandas.Series,
    forward_selection_col: str,
    option_1_rating_col: str,
    option_2_rating_col: str
) -> object:
    if row[forward_selection_col] == "option_1":
        return option_1_rating_col

    if row[forward_selection_col] == "option_2":
        return option_2_rating_col

    return None


def unique_columns(cols: list[str]) -> list[str]:
    unique_cols = []
    for col in cols:
        if col not in unique_cols:
            unique_cols.append(col)
    return unique_cols
