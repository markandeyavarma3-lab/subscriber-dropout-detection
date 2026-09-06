-- ===========================================================================
--  Subscriber Dropout Detection - Data Warehouse Demo
--  Open src/data/warehouse.db in DB Browser for SQLite, go to "Execute SQL",
--  and run these one block at a time (highlight a block, press Cmd+Return).
--
--  Every query here is fast. Anything slow is pre-computed - see Query 1.
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
--    Note the subscriber_id: a 44-character base64 SHA256 hash. The raw data
--    is anonymised by KKBox - no names, no emails, nothing identifying.
-- ---------------------------------------------------------------------------
SELECT * FROM subscribers LIMIT 10;


-- ---------------------------------------------------------------------------
-- 3. THE EVENT LOG
--    This is what makes the design work. Instead of one row per subscriber
--    with everything pre-aggregated, we store immutable timestamped events.
--    That lets us ask "what did this subscriber look like on 1 Nov 2016?"
--    which is impossible with a pre-aggregated table.
-- ---------------------------------------------------------------------------
SELECT subscriber_id, event_type, plan_type,
       ROUND(monthly_fee, 2) AS monthly_fee,
       is_auto_renew_enabled, occurred_at
FROM subscription_events
LIMIT 15;


-- ---------------------------------------------------------------------------
-- 4. ONE SUBSCRIBER'S COMPLETE HISTORY
--    Everything we know about a single person, assembled from three tables.
--    Watch the story: signs up March 2016, renews every month at 151.19,
--    then we have their daily listening from October onward.
--    This is the "360 degree view" the warehouse exists to produce.
-- ---------------------------------------------------------------------------
SELECT 'event'   AS record_type, occurred_at, event_type AS detail,
       ROUND(monthly_fee, 2) AS amount
FROM subscription_events
WHERE subscriber_id = '++/9R3sX37CjxbY/AaGvbwr3QkwElKBCtSvVzhCBDOk='
UNION ALL
SELECT 'payment', occurred_at, status, ROUND(amount, 2)
FROM payments
WHERE subscriber_id = '++/9R3sX37CjxbY/AaGvbwr3QkwElKBCtSvVzhCBDOk='
UNION ALL
SELECT 'session', occurred_at, 'listened', ROUND(duration_minutes, 1)
FROM sessions
WHERE subscriber_id = '++/9R3sX37CjxbY/AaGvbwr3QkwElKBCtSvVzhCBDOk='
ORDER BY occurred_at
LIMIT 40;


-- ---------------------------------------------------------------------------
-- 5. DATA CLEANING WE ACTUALLY DID
--    The raw export was not clean. These are real problems we found and fixed.
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
       COUNT(*)                                    AS events,
       ROUND(AVG(monthly_fee), 2)                  AS avg_monthly_fee,
       SUM(is_auto_renew_enabled)                  AS with_auto_renew
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
--    registered_via is an opaque integer in the source. We carry it through as
--    via_<n> rather than inventing channel names the data does not support.
-- ---------------------------------------------------------------------------
SELECT acquisition_channel, COUNT(*) AS subscribers
FROM subscribers
GROUP BY acquisition_channel
ORDER BY subscribers DESC
LIMIT 10;
