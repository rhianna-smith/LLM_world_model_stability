import pandas
import statsmodels.formula.api as smf
from omegaconf import DictConfig

def fit_clustered_ols(
    df: pandas.DataFrame,
    formula: str,
    cluster_col: str
) -> object:
    return smf.ols(
        formula=formula,
        data=df,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": df[cluster_col],
        },
    )


def fit_clustered_logit(
    df: pandas.DataFrame,
    formula: str,
    cluster_col: str
) -> object:
    return smf.logit(
        formula=formula,
        data=df,
    ).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={
            "groups": df[cluster_col],
        },
    )


def build_formula(
    outcome: str,
    predictor_cols: list[str],
    model_col: str,
    include_model_interactions: bool
) -> str:
    predictor_terms = [quote_formula_col(col) for col in predictor_cols]
    model_term = categorical_formula_col(model_col)
    if include_model_interactions:
        predictor_terms = [
            "%s * %s" % (
                term,
                model_term,
            )
            for term in predictor_terms
        ]
    else:
        predictor_terms.append(model_term)

    return "%s ~ %s" % (
        quote_formula_col(outcome),
        " + ".join(predictor_terms),
    )


def build_categorical_formula(
    outcome: str,
    predictor_cols: list[str],
    model_col: str,
    include_model_interactions: bool
) -> str:
    predictor_terms = [categorical_formula_col(col) for col in predictor_cols]
    model_term = categorical_formula_col(model_col)
    if include_model_interactions:
        predictor_terms = [
            "%s * %s" % (
                term,
                model_term,
            )
            for term in predictor_terms
        ]
    else:
        predictor_terms.append(model_term)

    return "%s ~ %s" % (
        quote_formula_col(outcome),
        " + ".join(predictor_terms),
    )


def categorical_formula_col(col: str) -> str:
    return "C(%s)" % quote_formula_col(col)


def quote_formula_col(col: str) -> str:
    return "Q(\"%s\")" % col


def fit_configured_models(df: pandas.DataFrame, cfg: DictConfig) -> dict[str, object]:
    models = {}

    for outcome in cfg.analysis.lpm_outcomes:
        formula = build_formula(
            outcome,
            list(cfg.analysis.predictor_cols),
            cfg.analysis.judgment_model_col,
            cfg.analysis.include_model_interactions,
        )
        models["ordinal_lpm_%s" % outcome] = fit_clustered_ols(
            df,
            formula,
            cfg.analysis.cluster_col,
        )
        categorical_formula = build_categorical_formula(
            outcome,
            list(cfg.analysis.categorical_predictor_cols),
            cfg.analysis.judgment_model_col,
            cfg.analysis.include_model_interactions,
        )
        models["categorical_lpm_%s" % outcome] = fit_clustered_ols(
            df,
            categorical_formula,
            cfg.analysis.cluster_col,
        )

    for outcome in cfg.analysis.linear_outcomes:
        formula = build_formula(
            outcome,
            list(cfg.analysis.predictor_cols),
            cfg.analysis.judgment_model_col,
            cfg.analysis.include_model_interactions,
        )
        models["ordinal_linear_%s" % outcome] = fit_clustered_ols(
            df,
            formula,
            cfg.analysis.cluster_col,
        )
        categorical_formula = build_categorical_formula(
            outcome,
            list(cfg.analysis.categorical_predictor_cols),
            cfg.analysis.judgment_model_col,
            cfg.analysis.include_model_interactions,
        )
        models["categorical_linear_%s" % outcome] = fit_clustered_ols(
            df,
            categorical_formula,
            cfg.analysis.cluster_col,
        )

    for outcome in cfg.analysis.logit_outcomes:
        formula = build_formula(
            outcome,
            list(cfg.analysis.predictor_cols),
            cfg.analysis.judgment_model_col,
            cfg.analysis.include_model_interactions,
        )
        models["ordinal_logit_%s" % outcome] = fit_clustered_logit(
            df,
            formula,
            cfg.analysis.cluster_col,
        )
        categorical_formula = build_categorical_formula(
            outcome,
            list(cfg.analysis.categorical_predictor_cols),
            cfg.analysis.judgment_model_col,
            cfg.analysis.include_model_interactions,
        )
        models["categorical_logit_%s" % outcome] = fit_clustered_logit(
            df,
            categorical_formula,
            cfg.analysis.cluster_col,
        )

    return models
