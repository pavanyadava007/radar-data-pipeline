-- Leakage check: how many measurements contribute segments to more than one split?
SELECT 'paper_split' AS split_kind,
       SUM(n_splits > 1) AS measurements_in_multiple_splits,
       COUNT(*)          AS measurements
FROM (SELECT measurement_id, COUNT(DISTINCT paper_split) AS n_splits FROM segments GROUP BY measurement_id)
UNION ALL
SELECT 'split_grouped',
       SUM(n_splits > 1),
       COUNT(*)
FROM (SELECT measurement_id, COUNT(DISTINCT split_grouped) AS n_splits FROM segments GROUP BY measurement_id);
