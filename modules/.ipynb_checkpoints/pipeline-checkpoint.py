import os
import concurrent.futures
import pandas
import tqdm
from omegaconf import DictConfig, OmegaConf
from modules.dataset import load_dataset
from modules.prompts import INSTRUCTION_TEMPLATE, REVERSE_INSTRUCTION_TEMPLATE
from modules.schemas import LLMAnswer
from modules.text import map_answer_to_option, map_answer_to_reverse_option, normalize_text
from typing import Callable, Protocol

class LLMClient(Protocol):
    def call(self, prompt: str, instruction_template: str) -> LLMAnswer:
        ...


def run_experiment(cfg: DictConfig, llm_client: LLMClient) -> None:
    output_dir = os.path.join(cfg.paths.output_root, cfg.model.output_name)
    os.makedirs(output_dir, exist_ok=True)
    save_resolved_config(cfg, output_dir)

    df = load_dataset(cfg.paths.dataset_path)

    forward_responses = run_prompt_stage(
        df=df,
        llm_client=llm_client,
        prompt_col="forward_prompt",
        instruction_template=INSTRUCTION_TEMPLATE,
        save_every=cfg.run.save_every,
        max_workers=cfg.run.max_workers,
        save_path=os.path.join(output_dir, "forward_responses_progress.csv"),
    )
    forward_responses.to_csv(os.path.join(output_dir, "forward_responses.csv"), index=False)

    experiment_df = run_prompt_stage(
        df=forward_responses,
        llm_client=llm_client,
        prompt_col="forward_swapped_prompt",
        instruction_template=INSTRUCTION_TEMPLATE,
        save_every=cfg.run.save_every,
        max_workers=cfg.run.max_workers,
        save_path=os.path.join(output_dir, "forward_swapped_responses_progress.csv"),
    )
    experiment_df.to_csv(
        os.path.join(output_dir, "forward_and_swapped_responses.csv"),
        index=False,
    )

    experiment_df = score_forward_stage(experiment_df)
    experiment_df.to_csv(os.path.join(output_dir, "forward_commutativity.csv"), index=False)
    print(experiment_df["forward_commutativity_score"].value_counts())

    reverse_df = build_reverse_stage_input(experiment_df)
    reverse_df = run_prompt_stage(
        df=reverse_df,
        llm_client=llm_client,
        prompt_col="reverse_prompt",
        instruction_template=REVERSE_INSTRUCTION_TEMPLATE,
        save_every=cfg.run.save_every,
        max_workers=cfg.run.max_workers,
        save_path=os.path.join(output_dir, "reverse_responses_progress.csv"),
    )

    reverse_df["reverse_swapped_prompt"] = progress_apply(
        reverse_df,
        select_reverse_swapped_prompt,
        "Selecting reverse swapped prompts",
    )
    reverse_df = run_prompt_stage(
        df=reverse_df,
        llm_client=llm_client,
        prompt_col="reverse_swapped_prompt",
        instruction_template=REVERSE_INSTRUCTION_TEMPLATE,
        save_every=cfg.run.save_every,
        max_workers=cfg.run.max_workers,
        save_path=os.path.join(output_dir, "reverse_swapped_responses_progress.csv"),
    )
    reverse_df.to_csv(os.path.join(output_dir, "reverse_responses.csv"), index=False)

    reverse_df = score_reverse_stage(reverse_df)
    print(reverse_df["reverse_commutativity_score"].value_counts())
    print(reverse_df["implication_score"].value_counts())

    final_df = build_final_results(experiment_df, reverse_df)
    write_final_outputs(final_df, output_dir)


def save_resolved_config(cfg: DictConfig, output_dir: str) -> None:
    config_path = os.path.join(output_dir, "resolved_config.yaml")
    with open(config_path, "w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(cfg, resolve=True))


def run_prompt_stage(
    df: pandas.DataFrame,
    llm_client: LLMClient,
    prompt_col: str,
    instruction_template: str,
    save_every: int,
    max_workers: int,
    save_path: str,
    skip_existing: bool = True
) -> pandas.DataFrame:
    df = df.copy()
    output_col = "%s_llm_output" % prompt_col
    final_col = "%s_final_answer" % prompt_col
    reason_col = "%s_reason" % prompt_col
    success_col = "%s_parse_success" % prompt_col
    error_col = "%s_parse_error" % prompt_col
    norm_col = "%s_final_answer_normalized" % prompt_col

    for col in [
        output_col,
        final_col,
        reason_col,
        success_col,
        error_col,
        norm_col,
    ]:
        if col not in df.columns:
            df[col] = None

    prompt_jobs = []
    row_iterator = tqdm.tqdm(
        df.iterrows(),
        total=len(df),
        desc="Preparing prompts for %s" % prompt_col,
    )
    for idx, row in row_iterator:
        prompt = row[prompt_col]

        if skip_existing:
            row_complete = (
                pandas.notna(row[output_col])
                and pandas.notna(row[final_col])
                and pandas.notna(row[norm_col])
                and row[success_col]
            )
            if row_complete:
                continue

        if prompt is None or pandas.isna(prompt):
            df.at[idx, output_col] = None
            df.at[idx, final_col] = None
            df.at[idx, reason_col] = None
            df.at[idx, success_col] = False
            df.at[idx, error_col] = "Prompt is null"
            df.at[idx, norm_col] = None
        else:
            prompt_jobs.append((idx, prompt))

    completed_since_save = 0
    max_workers = max(1, int(max_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                llm_client.call,
                prompt,
                instruction_template,
            ): idx
            for idx, prompt in prompt_jobs
        }
        progress_bar = tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Calling LLM for %s" % prompt_col,
        )
        for future in progress_bar:
            idx = futures[future]
            answer = future.result()
            df.at[idx, output_col] = answer.model_dump_json()
            df.at[idx, final_col] = answer.final_answer
            df.at[idx, reason_col] = answer.reason
            df.at[idx, success_col] = True
            df.at[idx, error_col] = None
            df.at[idx, norm_col] = normalize_text(answer.final_answer)

            completed_since_save += 1
            if completed_since_save >= save_every:
                df.to_csv(save_path, index=False)
                print("Saved progress to %s at row %s" % (save_path, idx))
                completed_since_save = 0

    df.to_csv(save_path, index=False)
    print("Final save to %s" % save_path)
    return df


def score_forward_stage(experiment_df: pandas.DataFrame) -> pandas.DataFrame:
    experiment_df = experiment_df.copy()
    experiment_df["forward_selected_option"] = progress_apply(
        experiment_df,
        lambda row: map_answer_to_option(
            row["forward_prompt_final_answer"],
            row["option_1"],
            row["option_2"],
        ),
        "Scoring forward prompts",
    )
    experiment_df["forward_swapped_selected_option_raw"] = progress_apply(
        experiment_df,
        lambda row: map_answer_to_option(
            row["forward_swapped_prompt_final_answer"],
            row["option_1_swapped"],
            row["option_2_swapped"],
        ),
        "Scoring forward swapped prompts",
    )
    experiment_df["forward_swapped_selected_option"] = progress_apply(
        experiment_df,
        convert_forward_swapped_back,
        "Converting forward swapped selections",
    )
    experiment_df["forward_selected_option_text"] = progress_apply(
        experiment_df,
        lambda row: get_selected_option_text(row, "forward_selected_option"),
        "Selecting forward option text",
    )
    experiment_df["forward_swapped_selected_option_text"] = progress_apply(
        experiment_df,
        lambda row: get_selected_option_text(row, "forward_swapped_selected_option"),
        "Selecting forward swapped option text",
    )
    experiment_df["forward_commutativity_score"] = progress_apply(
        experiment_df,
        score_forward_commutativity,
        "Scoring forward commutativity",
    )
    return experiment_df


def progress_apply(
    df: pandas.DataFrame,
    function: Callable[[pandas.Series], object],
    description: str
) -> pandas.Series:
    values = []
    indexes = []
    row_iterator = tqdm.tqdm(
        df.iterrows(),
        total=len(df),
        desc=description,
    )
    for idx, row in row_iterator:
        indexes.append(idx)
        values.append(function(row))

    return pandas.Series(values, index=indexes)


def convert_forward_swapped_back(row: pandas.Series) -> str:
    if row["forward_swapped_selected_option_raw"] == "option_1":
        return "option_2"

    if row["forward_swapped_selected_option_raw"] == "option_2":
        return "option_1"

    return "unclear"


def get_selected_option_text(row: pandas.Series, selected_col: str) -> object:
    if row[selected_col] == "option_1":
        return row["option_1"]

    if row[selected_col] == "option_2":
        return row["option_2"]

    return None


def score_forward_commutativity(row: pandas.Series) -> object:
    if row["forward_selected_option"] in [None, "unclear"]:
        return None

    if row["forward_swapped_selected_option"] in [None, "unclear"]:
        return None

    return int(row["forward_selected_option"] == row["forward_swapped_selected_option"])


def build_reverse_stage_input(experiment_df: pandas.DataFrame) -> pandas.DataFrame:
    reverse_df = experiment_df[
        experiment_df["forward_commutativity_score"] == 1
    ].copy()
    reverse_df["reverse_prompt"] = progress_apply(
        reverse_df,
        select_reverse_prompt,
        "Selecting reverse prompts",
    )
    reverse_df["reverse_prompt_selected"] = progress_apply(
        reverse_df,
        select_reverse_prompt_type,
        "Selecting reverse prompt types",
    )
    return reverse_df


def select_reverse_prompt(row: pandas.Series) -> object:
    if row["forward_selected_option"] == "option_1":
        return row["reverse_potential_prompt1"]

    if row["forward_selected_option"] == "option_2":
        return row["reverse_potential_prompt2"]

    return None


def select_reverse_prompt_type(row: pandas.Series) -> object:
    if row["forward_selected_option"] == "option_1":
        return "reverse_potential_prompt1"

    if row["forward_selected_option"] == "option_2":
        return "reverse_potential_prompt2"

    return None


def select_reverse_swapped_prompt(row: pandas.Series) -> object:
    if row["forward_selected_option"] == "option_1":
        return row["reverse_potential_swapped_prompt1"]

    if row["forward_selected_option"] == "option_2":
        return row["reverse_potential_swapped_prompt2"]

    return None


def score_reverse_stage(reverse_df: pandas.DataFrame) -> pandas.DataFrame:
    reverse_df = reverse_df.copy()
    reverse_df["reverse_selected_option"] = progress_apply(
        reverse_df,
        lambda row: map_answer_to_reverse_option(
            row["reverse_prompt_final_answer"],
            row["reverse_option_1"],
            row["reverse_option_2"],
        ),
        "Scoring reverse prompts",
    )
    reverse_df["reverse_swapped_selected_option_raw"] = progress_apply(
        reverse_df,
        lambda row: map_answer_to_reverse_option(
            row["reverse_swapped_prompt_final_answer"],
            row["reverse_option_2"],
            row["reverse_option_1"],
        ),
        "Scoring reverse swapped prompts",
    )
    reverse_df["reverse_swapped_selected_option"] = progress_apply(
        reverse_df,
        convert_reverse_swapped_back,
        "Converting reverse swapped selections",
    )
    reverse_df["reverse_commutativity_score"] = progress_apply(
        reverse_df,
        score_reverse_commutativity,
        "Scoring reverse commutativity",
    )
    reverse_df["implication_score"] = progress_apply(
        reverse_df,
        check_implication,
        "Scoring implication",
    )
    return reverse_df


def convert_reverse_swapped_back(row: pandas.Series) -> str:
    if row["reverse_swapped_selected_option_raw"] == "reverse_option_1":
        return "reverse_option_2"

    if row["reverse_swapped_selected_option_raw"] == "reverse_option_2":
        return "reverse_option_1"

    return "unclear"


def score_reverse_commutativity(row: pandas.Series) -> object:
    if row["reverse_selected_option"] in [None, "unclear"]:
        return None

    if row["reverse_swapped_selected_option"] in [None, "unclear"]:
        return None

    return int(row["reverse_selected_option"] == row["reverse_swapped_selected_option"])


def check_implication(row: pandas.Series) -> int:
    if row["reverse_commutativity_score"] == 0:
        return 0

    if row["forward_selected_option"] == "option_1":
        return int(
            row["reverse_selected_option"]
            == row["reverse_prompt1_expected_consistent_answer"]
        )

    if row["forward_selected_option"] == "option_2":
        return int(
            row["reverse_selected_option"]
            == row["reverse_prompt2_expected_consistent_answer"]
        )

    return 0


def build_final_results(
    experiment_df: pandas.DataFrame,
    reverse_df: pandas.DataFrame
) -> pandas.DataFrame:
    reverse_related_cols = [
        "base_experiment_ID",
        "reverse_prompt",
        "reverse_prompt_selected",
        "reverse_prompt_llm_output",
        "reverse_prompt_final_answer",
        "reverse_prompt_reason",
        "reverse_prompt_parse_success",
        "reverse_prompt_parse_error",
        "reverse_prompt_final_answer_normalized",
        "reverse_selected_option",
        "reverse_swapped_prompt",
        "reverse_swapped_prompt_llm_output",
        "reverse_swapped_prompt_final_answer",
        "reverse_swapped_prompt_reason",
        "reverse_swapped_prompt_parse_success",
        "reverse_swapped_prompt_parse_error",
        "reverse_swapped_prompt_final_answer_normalized",
        "reverse_swapped_selected_option_raw",
        "reverse_swapped_selected_option",
        "reverse_commutativity_score",
        "implication_score",
    ]
    return experiment_df.merge(
        reverse_df[reverse_related_cols],
        on="base_experiment_ID",
        how="left",
    )


def write_final_outputs(final_df: pandas.DataFrame, output_dir: str) -> None:
    unclear_forward_commutativity_df = final_df[
        final_df["forward_commutativity_score"].isna()
    ].copy()
    final_df = final_df[
        final_df["forward_commutativity_score"].notna()
    ].copy()
    print(final_df["forward_commutativity_score"].value_counts())

    failed_forward_mask = final_df["forward_commutativity_score"] == 0
    final_df.loc[failed_forward_mask, "reverse_commutativity_score"] = 0
    final_df.loc[failed_forward_mask, "implication_score"] = 0
    print(final_df["reverse_commutativity_score"].value_counts())

    unclear_reverse_commutativity_df = final_df[
        final_df["reverse_commutativity_score"].isna()
    ].copy()
    final_df = final_df[
        final_df["reverse_commutativity_score"].notna()
    ].copy()
    print(final_df["reverse_commutativity_score"].value_counts())

    failed_reverse_mask = final_df["reverse_commutativity_score"] == 0
    final_df.loc[failed_reverse_mask, "implication_score"] = 0
    print(final_df["implication_score"].value_counts())

    unclear_implication_df = final_df[
        final_df["implication_score"].isna()
    ].copy()
    print(final_df["implication_score"].value_counts())

    score_cols = [
        "forward_commutativity_score",
        "reverse_commutativity_score",
        "implication_score",
    ]
    final_df["total_consistency_score"] = final_df[score_cols].sum(axis=1)
    final_df["is_fully_consistent"] = (
        final_df["total_consistency_score"] == 3
    ).astype(int)

    final_df.to_csv(os.path.join(output_dir, "final_experiment_llm_results.csv"), index=False)
    unclear_forward_commutativity_df.to_csv(
        os.path.join(output_dir, "unclear_forward_commutativity.csv"),
        index=False,
    )
    unclear_reverse_commutativity_df.to_csv(
        os.path.join(output_dir, "unclear_reverse_commutativity.csv"),
        index=False,
    )
    unclear_implication_df.to_csv(
        os.path.join(output_dir, "unclear_implication.csv"),
        index=False,
    )

    analysis_df = final_df[
        [
            "base_experiment_ID",
            "forward_prompt",
            "forward_prompt_final_answer",
            "forward_selected_option",
            "forward_swapped_prompt",
            "forward_swapped_prompt_final_answer",
            "forward_swapped_selected_option",
            "reverse_prompt",
            "reverse_prompt_final_answer",
            "reverse_selected_option",
            "reverse_swapped_prompt",
            "reverse_swapped_prompt_final_answer",
            "reverse_swapped_selected_option",
            "forward_commutativity_score",
            "reverse_commutativity_score",
            "implication_score",
            "total_consistency_score",
            "is_fully_consistent",
        ]
    ].copy()
    print(final_df["total_consistency_score"].value_counts().sort_index())
    analysis_df.to_csv(
        os.path.join(output_dir, "final_experiment_llm_summary_results.csv"),
        index=False,
    )
