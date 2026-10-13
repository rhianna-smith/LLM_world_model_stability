import os
import pandas
from rhianna_thesis.analysis_tables import dataframe_to_markdown
from rhianna_thesis.console import print_subbanner


def build_marginal_predictions(
    models: dict[str, object],
    sample_dfs: dict[str, pandas.DataFrame],
    cfg: object,
) -> pandas.DataFrame:
    prediction_grids = []
    for model_name, model in models.items():
        sample_name = model_name.split("__")[0]
        grid_df = build_prediction_grid(sample_dfs[sample_name], cfg)
        grid_df["predicted_value"] = model.predict(grid_df)
        grid_df["model"] = model_name
        grid_df["sample"] = sample_name
        prediction_grids.append(grid_df)

    return pandas.concat(prediction_grids, ignore_index=True)


def build_prediction_grid(
    analysis_df: pandas.DataFrame,
    cfg: object,
) -> pandas.DataFrame:
    rows = []
    judgment_models = sorted(analysis_df[cfg.analysis.judgment_model_col].unique())
    moderator_values = sorted(analysis_df[cfg.analysis.moderator_col].unique())
    cluster_reference = analysis_df[cfg.analysis.cluster_col].iloc[0]
    forward_selection_reference = analysis_df[cfg.analysis.forward_selection_col].iloc[0]
    source_reference = analysis_df[cfg.analysis.selected_rating_source_col].iloc[0]
    rating_values = [
        rating_value
        for rating_value in cfg.analysis.marginal_rating_values
        if rating_value in set(analysis_df[cfg.analysis.selected_rating_col])
    ]

    for judgment_model in judgment_models:
        for moderator_value in moderator_values:
            for rating_value in rating_values:
                rows.append(
                    {
                        cfg.analysis.judgment_model_col: judgment_model,
                        cfg.analysis.moderator_col: moderator_value,
                        cfg.analysis.selected_rating_col: rating_value,
                        cfg.analysis.cluster_col: cluster_reference,
                        cfg.analysis.forward_selection_col: forward_selection_reference,
                        cfg.analysis.selected_rating_source_col: source_reference,
                    }
                )

    return pandas.DataFrame(rows)


def write_marginal_predictions(
    marginal_df: pandas.DataFrame,
    output_dir: str,
) -> None:
    csv_path = os.path.join(output_dir, "marginal_predictions_by_moderator.csv")
    markdown_path = os.path.join(output_dir, "marginal_predictions_by_moderator.md")
    marginal_df.to_csv(csv_path, index=False)
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write(dataframe_to_markdown(marginal_df, format_numeric=False))


def print_marginal_predictions(marginal_df: pandas.DataFrame) -> None:
    print_subbanner("Marginal Predictions By Judgment Model And Moderator")
    print(dataframe_to_markdown(marginal_df, format_numeric=False))
