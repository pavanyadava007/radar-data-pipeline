-- Segments, duration and range span per measurement (top 20 by size).
SELECT m.measurement_id,
       m.label,
       m.n_segments,
       ROUND(m.duration_s, 1)                       AS duration_s,
       ROUND(m.n_segments / NULLIF(m.duration_s, 0), 2) AS segments_per_s,
       ROUND(MIN(s.range_m), 1)                     AS range_min_m,
       ROUND(MAX(s.range_m), 1)                     AS range_max_m,
       m.split_grouped
FROM measurements m JOIN segments s USING (measurement_id)
GROUP BY m.measurement_id
ORDER BY m.n_segments DESC
LIMIT 20;
