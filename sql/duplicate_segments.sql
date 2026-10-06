-- Exact duplicate segments (identical sha1 of the complex samples): per pair of measurements,
-- with label conflicts and whether the copies sit in different paper splits (train/test leakage).
WITH d AS (
    SELECT sha1 FROM segments GROUP BY sha1 HAVING COUNT(*) > 1
)
SELECT a.measurement_id AS meas_a,
       ma.label         AS label_a,
       b.measurement_id AS meas_b,
       mb.label         AS label_b,
       COUNT(*)         AS duplicate_pairs,
       SUM(a.paper_split <> b.paper_split) AS pairs_across_paper_splits,
       ma.label <> mb.label AS label_conflict
FROM segments a
JOIN segments b ON a.sha1 = b.sha1 AND a.segment_id < b.segment_id
JOIN measurements ma ON ma.measurement_id = a.measurement_id
JOIN measurements mb ON mb.measurement_id = b.measurement_id
WHERE a.sha1 IN (SELECT sha1 FROM d)
GROUP BY a.measurement_id, b.measurement_id
ORDER BY duplicate_pairs DESC;
