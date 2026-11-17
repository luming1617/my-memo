# Multilingual eval report: s1-mock

- timestamp: 2026-11-17T02:06:53.489872+00:00
- git commit: efdcb485642f9ac5bb03f29c94c1f01ae7057d81
- capabilities: {"spacy_model": true, "fastembed_bm25_cached": true}
- mode: mock

## Per-class summary

| class | cases | fact_cov | lang_match | hit_rate | mrr | entity_boost | guard_viol | isolation | pair_sim | pair_co_return |
|---|---|---|---|---|---|---|---|---|---|---|
| zh_extraction | 2 | 1.0 | 1.0 | 1.0 | 1.0 | - | - | - | - | - |
| en_extraction | 2 | 1.0 | 1.0 | 1.0 | 1.0 | - | - | - | - | - |
| mixed_extraction | 1 | 1.0 | - | - | - | - | - | - | - | - |
| cross_lingual_recall | 2 | - | - | 0.0 | - | - | - | - | - | - |
| lexical_precision_zh | 2 | - | - | 1.0 | 1.0 | - | - | - | - | - |
| entity_boost_zh | 1 | - | - | 1.0 | 1.0 | 0.0 | - | - | - | - |
| entity_alias_guard | 1 | - | - | - | - | - | 0 | - | - | - |
| near_duplicate_translation | 1 | - | - | 1.0 | 1.0 | - | - | - | 0.0 | 1.0 |
| near_duplicate_negation | 1 | - | - | 1.0 | 1.0 | - | - | - | 0.5729 | 2.0 |
| near_duplicate_number | 1 | - | - | 1.0 | 1.0 | - | - | - | 0.9333 | 2.0 |
| multi_tenant | 1 | - | - | 1.0 | 1.0 | - | - | 1.0 | - | - |

## Explain path statistics

| class | queries | bm25_active | entity_active | mean_semantic | mean_bm25 | mean_entity | mean_max_possible |
|---|---|---|---|---|---|---|---|
| zh_extraction | 1 | 0 | 0 | 0.3904 | 0.0 | 0.0 | 1.0 |
| en_extraction | 1 | 1 | 0 | 0.1593 | 0.0528 | 0.0 | 2.0 |
| mixed_extraction | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cross_lingual_recall | 2 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| lexical_precision_zh | 2 | 1 | 0 | 0.3816 | 0.03 | 0.0 | 1.5 |
| entity_boost_zh | 1 | 0 | 0 | 0.5345 | 0.0 | 0.0 | 1.0 |
| entity_alias_guard | 1 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| near_duplicate_translation | 1 | 0 | 0 | 0.4201 | 0.0 | 0.0 | 1.0 |
| near_duplicate_negation | 1 | 0 | 0 | 0.4638 | 0.0 | 0.0 | 1.0 |
| near_duplicate_number | 1 | 1 | 0 | 0.4472 | 0.0809 | 0.0 | 2.0 |
| multi_tenant | 1 | 0 | 0 | 0.4201 | 0.0 | 0.0 | 1.0 |
