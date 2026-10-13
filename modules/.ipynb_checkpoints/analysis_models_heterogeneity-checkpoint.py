import pandas
import statsmodels.formula.api as smf
from omegaconf import DictConfig


def fit_clustered_ols(
    df: pandas.DataFrame,
    formula: str,
    cluster_col: str,
) -> object:
    return smf.ols(formula=formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col]},
    )


def fit_clustered_logit(
    df: pandas.DataFrame,
    formula: str,
    cluster_col: str,
) -> object:
    return smf.logit(formula=formula, data=df).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col]},
    )


def build_heterogeneity_formula(
    outcome: str,
    predictor_cols: list[str],
    model_col: str,
    moderator_col: str,
    include_model_interactions: bool,
    predictors_are_categorical: bool,
) -> str:
    if predictors_are_categorical:
        predictor_terms = [categorical_formula_col(col) for col in predictor_cols]
    else:
        predictor_terms = [quote_formula_col(col) for col in predictor_cols]

    model_term = categorical_formula_col(model_col)
    moderator_term = categorical_formula_col(moderator_col)

    terms = []
    for predictor_term in predictor_terms:
        # Main slice-and-dice term: likelihood rating interacted with task attribute.
        terms.append("%s * %s" % (predictor_term, moderator_term))

        # Preserve the original analysis's option to let the rating effect differ by
        # judgment model, without creating a three-way interaction.
        if include_model_interactions:
            terms.append("%s * %s" % (predictor_term, model_term))
        else:
            terms.append(model_term)

    return "%s ~ %s" % (
        quote_formula_col(outcome),
        " + ".join(unique_terms(terms)),
    )


def categorical_formula_col(col: str) -> str:
    return "C(%s)" % quote_formula_col(col)


def quote_formula_col(col: str) -> str:
    return "Q(\"%s\")" % col


def unique_terms(terms: list[str]) -> list[str]:
    out = []
    for term in terms:
        if term not in out:
            out.append(term)
    return out


def fit_configured_models(df: pandas.DataFrame, cfg: DictConfig) -> dict[str, object]:
    models = {}
    moderator_col = cfg.analysis.moderator_col

    for outcome in cfg.analysis.lpm_outcomes:
        formula = build_heterogeneity_formula(
            outcome=outcome,
            predictor_cols=list(cfg.analysis.predictor_cols),
            model_col=cfg.analysis.judgment_model_col,
            moderator_col=moderator_col,
            include_model_interactions=cfg.analysis.include_model_interactions,
            predictors_are_categorical=False,
        )
        models["ordinal_lpm_%s_by_%s" % (outcome, moderator_col)] = fit_clustered_ols(
            df, formula, cfg.analysis.cluster_col
        )

        categorical_formula = build_heterogeneity_formula(
            outcome=outcome,
            predictor_cols=list(cfg.analysis.categorical_predictor_cols),
            model_col=cfg.analysis.judgment_model_col,
            moderator_col=moderator_col,
            include_model_interactions=cfg.analysis.include_model_interactions,
            predictors_are_categorical=True,
        )
        models["categorical_lpm_%s_by_%s" % (outcome, moderator_col)] = fit_clustered_ols(
            df, categorical_formula, cfg.analysis.cluster_col
        )

    for outcome in cfg.analysis.linear_outcomes:
        formula = build_heterogeneity_formula(
            outcome=outcome,
            predictor_cols=list(cfg.analysis.predictor_cols),
            model_col=cfg.analysis.judgment_model_col,
            moderator_col=moderator_col,
            include_model_interactions=cfg.analysis.include_model_interactions,
            predictors_are_categorical=False,
        )
        models["ordinal_linear_%s_by_%s" % (outcome, moderator_col)] = fit_clustered_ols(
            df, formula, cfg.analysis.cluster_col
        )

        categorical_formula = build_heterogeneity_formula(
            outcome=outcome,
            predictor_cols=list(cfg.analysis.categorical_predictor_cols),
            model_col=cfg.analysis.judgment_model_col,
            moderator_col=moderator_col,
            include_model_interactions=cfg.analysis.include_model_interactions,
            predictors_are_categorical=True,
        )
        models["categorical_linear_%s_by_%s" % (outcome, moderator_col)] = fit_clustered_ols(
            df, categorical_formula, cfg.analysis.cluster_col
        )

    for outcome in cfg.analysis.logit_outcomes:
        formula = build_heterogeneity_formula(
            outcome=outcome,
            predictor_cols=list(cfg.analysis.predictor_cols),
            model_col=cfg.analysis.judgment_model_col,
            moderator_col=moderator_col,
            include_model_interactions=cfg.analysis.include_model_interactions,
            predictors_are_categorical=False,
        )
        models["ordinal_logit_%s_by_%s" % (outcome, moderator_col)] = fit_clustered_logit(
            df, formula, cfg.analysis.cluster_col
        )

        categorical_formula = build_heterogeneity_formula(
            outcome=outcome,
            predictor_cols=list(cfg.analysis.categorical_predictor_cols),
            model_col=cfg.analysis.judgment_model_col,
            moderator_col=moderator_col,
            include_model_interactions=cfg.analysis.include_model_interactions,
            predictors_are_categorical=True,
        )
        models["categorical_logit_%s_by_%s" % (outcome, moderator_col)] = fit_clustered_logit(
            df, categorical_formula, cfg.analysis.cluster_col
        )

    return models
