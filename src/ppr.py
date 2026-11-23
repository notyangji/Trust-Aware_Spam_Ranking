from src.rank import power_iteration
from src.seeds import make_personalization


def personalized_pagerank(M, seed_ids, **rank_options):
    """선택한 seed를 기준으로 Personalized PageRank 점수를 계산한다."""
    s = make_personalization(M.shape[0], seed_ids)
    return power_iteration(M, teleport=s, **rank_options)
