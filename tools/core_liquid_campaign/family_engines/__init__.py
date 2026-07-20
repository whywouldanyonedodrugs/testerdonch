from . import a1_compression, a2_context, a3_starter_retest, a4_tsmom, kda02b_adjudication

ENGINES = {
    "A4_TSMOM_V7": a4_tsmom,
    "A1_COMPRESSION_V2": a1_compression,
    "A2_PRIOR_HIGH_RS_CONTEXT_V1": a2_context,
    "A3_STARTER_RETEST_V3": a3_starter_retest,
    "KDA02B_SURVIVOR_ADJUDICATION_V1": kda02b_adjudication,
}

__all__ = ["ENGINES"]
