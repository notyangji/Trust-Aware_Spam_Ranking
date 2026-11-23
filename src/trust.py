from src.ppr import personalized_pagerank


def trustrank(M, seed_ids, **rank_options):
    """Trusted seed에 같은 비중을 주고 링크를 따라 신뢰 점수를 전파한다.

    M은 build_transition()이 만든 전이 행렬이다.
    seed_ids는 select_trusted_seeds()가 고른 원본 호스트 ID다.
    rank_options는 beta, tol, return_info 같은 공통 랭킹 옵션이다.
    호스트 ID 순서로 신뢰 점수를 반환하고, return_info=True이면 수렴 정보도 반환한다.
    """
    return personalized_pagerank(M, seed_ids, **rank_options)
