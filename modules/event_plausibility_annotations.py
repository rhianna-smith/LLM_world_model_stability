import os
import concurrent.futures
import pandas
import tqdm
from omegaconf import DictConfig, OmegaConf
from modules.annotation_prompts import EVENT_LIKELIHOOD_INSTRUCTIONS
from modules.schemas import EventLikelihoodAnnotation
from typing import Protocol

class AnnotationClient(Protocol):
    def call(
        self,
        prompt: str,
        instruction_template: str,
        response_schema: type[EventLikelihoodAnnotation] = EventLikelihoodAnnotation
    ) -> EventLikelihoodAnnotation:
        ...


def run_event_likelihood_annotations(
    cfg: DictConfig,
    llm_client: AnnotationClient
) -> None:
    os.makedirs(cfg.paths.annotation_output_dir, exist_ok=True)
    save_resolved_config(cfg, cfg.paths.annotation_output_dir)

    dataset_df = pandas.read_csv(cfg.paths.dataset_path)
    ratings_df = dataset_df[
        [
            "base_experiment_ID",
            "forward_answer_potential_statement1",
            "forward_answer_potential_statement2",
        ]
    ].copy()

    progress_path = os.path.join(
        cfg.paths.annotation_output_dir,
        cfg.event_likelihood.progress_filename,
    )
    ratings_df = annotate_statement_column(
        df=ratings_df,
        llm_client=llm_client,
        statement_col="forward_answer_potential_statement1",
        save_every=cfg.run.save_every,
        max_workers=cfg.run.max_workers,
        save_path=progress_path,
    )
    ratings_df = annotate_statement_column(
        df=ratings_df,
        llm_client=llm_client,
        statement_col="forward_answer_potential_statement2",
        save_every=cfg.run.save_every,
        max_workers=cfg.run.max_workers,
        save_path=progress_path,
    )
    ratings_df.to_csv(
        os.path.join(
            cfg.paths.annotation_output_dir,
            cfg.event_likelihood.final_filename,
        ),
        index=False,
    )


def save_resolved_config(cfg: DictConfig, output_dir: str) -> None:
    config_path = os.path.join(output_dir, "event_likelihood_resolved_config.yaml")
    with open(config_path, "w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(cfg, resolve=True))


def annotate_statement_column(
    df: pandas.DataFrame,
    llm_client: AnnotationClient,
    statement_col: str,
    save_every: int,
    max_workers: int,
    save_path: str,
    skip_existing: bool = True
) -> pandas.DataFrame:
    df = df.copy()
    output_col = "llm_rating_output_%s" % statement_col
    rating_col = "llm_rating_%s" % statement_col
    label_col = "llm_rating_label_%s" % statement_col
    reason_col = "llm_rating_reason_%s" % statement_col
    success_col = "llm_rating_parse_success_%s" % statement_col
    error_col = "llm_rating_parse_error_%s" % statement_col

    for col in [
        output_col,
        rating_col,
        label_col,
        reason_col,
        success_col,
        error_col,
    ]:
        if col not in df.columns:
            df[col] = None

    statement_jobs = collect_statement_jobs(
        df=df,
        statement_col=statement_col,
        output_col=output_col,
        rating_col=rating_col,
        success_col=success_col,
        skip_existing=skip_existing,
        error_col=error_col,
    )

    completed_since_save = 0
    max_workers = max(1, int(max_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                annotate_statement,
                llm_client,
                statement,
            ): idx
            for idx, statement in statement_jobs
        }
        progress_bar = tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Annotating %s" % statement_col,
        )
        for future in progress_bar:
            idx = futures[future]
            annotation = future.result()
            df.at[idx, output_col] = annotation.model_dump_json()
            df.at[idx, rating_col] = int(annotation.rating)
            df.at[idx, label_col] = annotation.rating.name
            df.at[idx, reason_col] = annotation.reason
            df.at[idx, success_col] = True
            df.at[idx, error_col] = None

            completed_since_save += 1
            if completed_since_save >= save_every:
                df.to_csv(save_path, index=False)
                print("Saved progress to %s at row %s" % (save_path, idx))
                completed_since_save = 0

    df.to_csv(save_path, index=False)
    print("Final save to %s" % save_path)
    return df


def collect_statement_jobs(
    df: pandas.DataFrame,
    statement_col: str,
    output_col: str,
    rating_col: str,
    success_col: str,
    skip_existing: bool,
    error_col: str
) -> list[tuple[object, str]]:
    statement_jobs = []
    row_iterator = tqdm.tqdm(
        df.iterrows(),
        total=len(df),
        desc="Preparing annotations for %s" % statement_col,
    )
    for idx, row in row_iterator:
        statement = row[statement_col]
        row_complete = (
            pandas.notna(row[output_col])
            and pandas.notna(row[rating_col])
            and row[success_col]
        )

        if skip_existing and row_complete:
            continue

        if statement is None or pandas.isna(statement):
            df.at[idx, output_col] = None
            df.at[idx, rating_col] = None
            df.at[idx, success_col] = False
            df.at[idx, error_col] = "Statement is null"
        else:
            statement_jobs.append((idx, statement))

    return statement_jobs


def annotate_statement(
    llm_client: AnnotationClient,
    statement: str
) -> EventLikelihoodAnnotation:
    return llm_client.call(
        "Statement to rate:\n%s" % statement,
        EVENT_LIKELIHOOD_INSTRUCTIONS,
        EventLikelihoodAnnotation,
    )
