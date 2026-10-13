import os
import concurrent.futures
import pandas
import tqdm
from omegaconf import DictConfig, OmegaConf
from rhianna_thesis.dataset import load_dataset
from rhianna_thesis.prompts import INSTRUCTION_TEMPLATE, REVERSE_INSTRUCTION_TEMPLATE
from rhianna_thesis.schemas import LLMAnswer
from rhianna_thesis.text import normalize_text
from rhianna_thesis.pipeline import (
    score_forward_stage,
    progress_apply,
    build_reverse_stage_input,
    select_reverse_swapped_prompt,
    score_reverse_stage,
    build_final_results,
    write_final_outputs,
)
from typing import Protocol


class LLMClient(Protocol):
    def call(self, prompt: str, instruction_template: str) -> LLMAnswer:
        ...


def run_context_experiment(cfg: DictConfig, llm_client: LLMClient) -> None:
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
    forward_responses.to_csv(
        os.path.join(output_dir, "forward_responses.csv"),
        index=False,
    )

    experiment_df = run_prompt_stage(
        df=forward_responses,
        llm_client=llm_client,
        prompt_col="forward_swapped_prompt",
        instruction_template=INSTRUCTION_TEMPLATE,
        save_every=cfg.run.save_every,
        max_workers=cfg.run.max_workers,
        save_path=os.path.join(
            output_dir,
            "forward_swapped_responses_progress.csv",
        ),
    )
    experiment_df.to_csv(
        os.path.join(output_dir, "forward_and_swapped_responses.csv"),
        index=False,
    )

    experiment_df = score_forward_stage(experiment_df)
    experiment_df.to_csv(
        os.path.join(output_dir, "forward_commutativity.csv"),
        index=False,
    )
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
        save_path=os.path.join(
            output_dir,
            "reverse_swapped_responses_progress.csv",
        ),
    )

    reverse_df.to_csv(
        os.path.join(output_dir, "reverse_responses.csv"),
        index=False,
    )

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
        supporter = row["supporter"]

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
            prompt = (
                "Supporting context:\n%s\n\n%s"
                % (supporter, prompt)
            )
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