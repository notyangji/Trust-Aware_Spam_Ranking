### Final test (SET2) 결과와 같은 조건의 val

기준: 같은 random_seed의 기본 TrustRank. test SPAM 비율 = 0.0594. * domain_high_pr은 val 결과를 보고 추가한 전략.

| group | method | seed_strategy | test PR-AUC | test sd | test ROC-AUC | test P@100 | test Δ vs 기본 TR | test ↑ | test ↓ | val PR-AUC | val Δ vs 기본 TR | val ↑ | val ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines | PPR (ac.uk·gov.uk seeds) | Random | 0.0842 | 0 | 0.6360 | 0.0760 | 0.0066 | 4 | 0 | 0.0913 | 0.0073 | 5 | 0 |
| baselines | Spam mass | Random | 0.0797 | 0.0038 | 0.6073 | 0.0772 | 0.0020 | 4 | 0 | 0.0854 | 0.0014 | 2 | 0 |
| baselines | TrustRank | Random | 0.0777 | 0.0030 | 0.5952 | 0.0772 | 0 | 0 | 0 | 0.0841 | 0 | 0 | 0 |
| baselines | Hop distance | Random | 0.0743 | 0.0029 | 0.5914 | 0.0863 | -0.0038 | 0 | 0 | 0.0834 | -0.0017 | 0 | 0 |
| baselines | PageRank | Random | 0.0735 | 0 | 0.5541 | 0.0865 | -0.0041 | 0 | 2 | 0.0739 | -0.0104 | 0 | 3 |
| baselines | In-degree | Random | 0.0725 | 0 | 0.5728 | 0.0779 | -0.0055 | 0 | 0 | 0.0857 | 0.0011 | 0 | 0 |
| baselines | PageRank (weighted) | Random | 0.0716 | 0 | 0.5533 | 0.0745 | -0.0060 | 0 | 3 | 0.0713 | -0.0133 | 0 | 4 |
| baselines | Inverse PageRank | Random | 0.0710 | 0 | 0.5506 | 0.0772 | -0.0076 | 0 | 0 | 0.0642 | -0.0228 | 0 | 3 |
| baselines | Random | Random | 0.0645 | 0.0066 | 0.4975 | 0.0680 | -0.0122 | 0 | 0 | 0.0645 | -0.0196 | 0 | 1 |
| selected | Spam mass | Trusted domain + top PR* | 0.0869 | 0 | 0.6516 | 0.0700 | 0.0093 | 5 | 0 | 0.0947 | 0.0109 | 5 | 0 |
| selected | Spam mass | Trusted domain | 0.0864 | 0.0006 | 0.6485 | 0.0754 | 0.0087 | 5 | 0 | 0.0921 | 0.0081 | 5 | 0 |
| selected | Spam mass | Top PageRank | 0.0854 | 0 | 0.6381 | 0.0700 | 0.0078 | 5 | 0 | 0.0932 | 0.0095 | 5 | 0 |
| selected | TrustRank | Trusted domain + top PR* | 0.0841 | 0 | 0.6367 | 0.0700 | 0.0064 | 4 | 0 | 0.0926 | 0.0087 | 5 | 0 |
| selected | TrustRank | Trusted domain | 0.0835 | 0.0005 | 0.6330 | 0.0754 | 0.0059 | 4 | 0 | 0.0899 | 0.0059 | 2 | 0 |
| selected | TrustRank | Top PageRank | 0.0827 | 0 | 0.6245 | 0.0700 | 0.0051 | 3 | 0 | 0.0914 | 0.0077 | 4 | 0 |
