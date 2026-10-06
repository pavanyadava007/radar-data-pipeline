-- Real-data quality flags per label: edge truncation (dataset label), integrity rule flags, low SNR, time gaps.
SELECT m.label,
       COUNT(*)                                  AS n_segments,
       SUM(s.edge_flag)                          AS edge_truncated,
       SUM(s.rule_flags > 0)                     AS any_rule_flag,
       ROUND(100.0 * SUM(s.rule_flags > 0) / COUNT(*), 2) AS rule_flag_pct,
       SUM(s.low_snr)                            AS low_snr,
       SUM(s.time_gap)                           AS time_gap_before
FROM segments s JOIN measurements m USING (measurement_id)
GROUP BY m.label
ORDER BY n_segments DESC;
