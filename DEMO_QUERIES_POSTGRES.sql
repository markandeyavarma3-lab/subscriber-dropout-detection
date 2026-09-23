-- ===========================================================================
--  Subscriber Dropout Detection - Data Warehouse Demo (PostgreSQL)
--  Open the `warehouse` connection in TablePlus, open a SQL tab (Cmd+E), paste
--  a block, select it, and run it (Cmd+Return).
--
--  Same eight stories as DEMO_QUERIES.sql, which stays as the SQLite fallback.
--  Two things differ, both because Postgres is stricter than SQLite:
--    * ROUND(x, 2) needs x::numeric - Postgres has no ROUND for a float with
--      a precision argument, where SQLite accepts anything.
--    * Booleans are real booleans, so they are counted with FILTER rather
--      than summed as if they were 0/1 integers.
-- ===========================================================================


-- ---------------------------------------------------------------------------
-- 1. WHAT IS IN HERE
--    82.8 million rows, built from 31.3 GB of raw CSV files.
-- ---------------------------------------------------------------------------
SELECT table_name, row_count, what_it_holds, source
FROM warehouse_summary
ORDER BY row_count DESC;


-- ---------------------------------------------------------------------------
-- 2. THE SUBSCRIBERS TABLE
--    subscriber_id is a 44-character base64 SHA256 hash. KKBox anonymised the
--    export - no names, no emails, nothing identifying.
-- ---------------------------------------------------------------------------
SELECT * FROM subscribers LIMIT 10;


-- ---------------------------------------------------------------------------
-- 3. THE EVENT LOG
--    Immutable, timestamped events rather than one pre-aggregated row per
--    subscriber. That is what lets us ask "what did this subscriber look like
--    on 1 Nov 2016?" - impossible once history has been summed away.
-- ---------------------------------------------------------------------------
SELECT subscriber_id, event_type, plan_type,
       ROUND(monthly_fee::numeric, 2) AS monthly_fee,
       is_auto_renew_enabled, occurred_at
FROM subscription_events
LIMIT 15;


-- ---------------------------------------------------------------------------
-- 4. ONE SUBSCRIBER'S COMPLETE HISTORY
--    Everything we know about one person, assembled from three tables. Signs
--    up March 2016, renews monthly at 151.19, and listens daily from October.
--    Fast because every table has an index on (subscriber_id, occurred_at).
-- ---------------------------------------------------------------------------
SELECT 'event' AS record_type, occurred_at, event_type AS detail,
       ROUND(monthly_fee::numeric, 2) AS amount
FROM subscription_events
WHERE subscriber_id = '++/9R3sX37CjxbY/AaGvbwr3QkwElKBCtSvVzhCBDOk='
UNION ALL
SELECT 'payment', occurred_at, status, ROUND(amount::numeric, 2)
FROM payments
WHERE subscriber_id = '++/9R3sX37CjxbY/AaGvbwr3QkwElKBCtSvVzhCBDOk='
UNION ALL
SELECT 'session', occurred_at, 'listened', ROUND(duration_minutes::numeric, 1)
FROM sessions
WHERE subscriber_id = '++/9R3sX37CjxbY/AaGvbwr3QkwElKBCtSvVzhCBDOk='
ORDER BY occurred_at
LIMIT 40;


-- ---------------------------------------------------------------------------
-- 5. DATA CLEANING WE ACTUALLY DID
--    The raw export was not clean. These are real problems found and fixed.
-- ---------------------------------------------------------------------------
SELECT 'Orphan rows removed'      AS cleaning_step,
       '2,656,043'                AS amount,
       'Transactions referencing 432,623 subscribers absent from the members file (18.3%). '
       || 'Left in, they would vanish silently from every query.' AS why_it_mattered
UNION ALL
SELECT 'ID column widened',
       '32 -> 64 chars',
       'Real IDs are 44 characters. SQLite ignores the limit; Postgres would have rejected every row.'
UNION ALL
SELECT 'Negative durations clipped',
       'to zero',
       'A known logging artefact. Dropping the rows would understate activity, which reads as churn risk.'
UNION ALL
SELECT 'Prices normalised',
       'to monthly rate',
       'Plans run 7 to 410 days. Comparing a 410-day payment to a 30-day one is meaningless.'
UNION ALL
SELECT 'Session window bounded',
       '392M -> 38.2M rows',
       'Full history is more than a 30-day feature window can use, and 7+ hours to write.';


-- ---------------------------------------------------------------------------
-- 6. PLAN MIX
--    Derived from payment_plan_days, since KKBox has no tier names.
-- ---------------------------------------------------------------------------
SELECT plan_type,
       COUNT(*)                                         AS events,
       ROUND(AVG(monthly_fee)::numeric, 2)              AS avg_monthly_fee,
       COUNT(*) FILTER (WHERE is_auto_renew_enabled)    AS with_auto_renew
FROM subscription_events
GROUP BY plan_type
ORDER BY events DESC;


-- ---------------------------------------------------------------------------
-- 7. LIFECYCLE EVENT BREAKDOWN
--    Cancellations are the churn signal the model learns to predict.
-- ---------------------------------------------------------------------------
SELECT event_type, COUNT(*) AS occurrences
FROM subscription_events
GROUP BY event_type
ORDER BY occurrences DESC;


-- ---------------------------------------------------------------------------
-- 8. HOW SUBSCRIBERS WERE ACQUIRED
--    registered_via is an opaque integer in the source. Carried through as
--    via_<n> rather than inventing channel names the data does not support.
-- ---------------------------------------------------------------------------
SELECT acquisition_channel, COUNT(*) AS subscribers
FROM subscribers
GROUP BY acquisition_channel
ORDER BY subscribers DESC
LIMIT 10;


-- ===========================================================================
--  EXTRA QUERIES - for when the examiner asks something unplanned.
-- ===========================================================================


-- ---------------------------------------------------------------------------
-- 9. HOW BIG IS IT?
--    Size on disk, indexes included. sessions alone is about 7.3 GB.
-- ---------------------------------------------------------------------------
SELECT relname AS table_name,
       pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;


-- ---------------------------------------------------------------------------
-- 10. WHY IS IT FAST? THE INDEXES
--     Every event table has an index on (subscriber_id, occurred_at): "this
--     subscriber's events before this date" is the question nearly every
--     feature asks.
-- ---------------------------------------------------------------------------
SELECT tablename, indexname, indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;


-- ---------------------------------------------------------------------------
-- 11. PROOF THE INDEX IS USED
--     Look for "Index Scan using ix_sessions_sub_time" and the execution time:
--     63 sessions found among 38 million rows in well under a millisecond.
-- ---------------------------------------------------------------------------
EXPLAIN ANALYZE
SELECT occurred_at, duration_minutes
FROM sessions
WHERE subscriber_id = '++/9R3sX37CjxbY/AaGvbwr3QkwElKBCtSvVzhCBDOk='
ORDER BY occurred_at;


-- ---------------------------------------------------------------------------
-- 12. CANCELLATIONS PER MONTH
--     A count, not a rate: it also grows as the subscriber base grows.
-- ---------------------------------------------------------------------------
SELECT date_trunc('month', occurred_at)::date AS month,
       COUNT(*)                               AS cancellations
FROM subscription_events
WHERE event_type = 'cancellation'
  AND occurred_at >= '2016-06-01'
GROUP BY 1
ORDER BY 1;
