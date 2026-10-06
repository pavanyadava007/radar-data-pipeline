-- Segments per class group under the paper (per-sample) split and the grouped (per-measurement) split.
SELECT m.class_group,
       SUM(s.paper_split = 1)   AS paper_train,
       SUM(s.paper_split = 2)   AS paper_val,
       SUM(s.paper_split = 3)   AS paper_test,
       SUM(s.split_grouped = 1) AS grouped_train,
       SUM(s.split_grouped = 2) AS grouped_val,
       SUM(s.split_grouped = 3) AS grouped_test,
       COUNT(*)                 AS total
FROM segments s JOIN measurements m USING (measurement_id)
GROUP BY m.class_group
ORDER BY total DESC;
