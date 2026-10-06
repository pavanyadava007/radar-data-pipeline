-- SNR (dB, peak centre-cell RD power over noise floor) and Doppler spread per label.
SELECT m.label,
       m.class_group,
       COUNT(*)                          AS n,
       ROUND(AVG(s.snr_db), 2)           AS snr_db_mean,
       ROUND(MIN(s.snr_db), 2)           AS snr_db_min,
       ROUND(MAX(s.snr_db), 2)           AS snr_db_max,
       ROUND(AVG(s.doppler_spread_hz), 1) AS doppler_spread_hz_mean,
       ROUND(AVG(s.md_bandwidth_hz), 1)   AS md_bandwidth_hz_mean
FROM segments s JOIN measurements m USING (measurement_id)
GROUP BY m.label
ORDER BY snr_db_mean DESC;
