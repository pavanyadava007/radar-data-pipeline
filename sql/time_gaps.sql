-- Time gaps (> 0.5 s between consecutive segments) per label: target left the field of view or was lost in noise.
SELECT m.label,
       COUNT(DISTINCT m.measurement_id)                 AS measurements,
       SUM(s.gap_before_s > 0.5)                        AS gaps_over_0_5s,
       ROUND(MAX(s.gap_before_s), 2)                    AS max_gap_s,
       ROUND(AVG(CASE WHEN s.gap_before_s <= 0.5 THEN s.gap_before_s END), 4) AS typical_dt_s
FROM segments s JOIN measurements m USING (measurement_id)
GROUP BY m.label
ORDER BY gaps_over_0_5s DESC;
