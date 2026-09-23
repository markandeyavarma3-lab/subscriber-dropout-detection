# Likely viva questions, with answers

Short answers in plain words. Learn the *idea* behind each one, not the wording. An examiner
who hears a memorised paragraph asks a follow-up, and the follow-up is where understanding
shows. Every number here comes from `src/models/artifacts/metrics.json` or the warehouse
itself.

---

## The problem and the data

**1. What problem does this solve?**
A subscription business loses revenue when subscribers leave. If you can predict who is
*about to* leave, you can offer them something (a discount, a better plan) while they're
still a customer. This project predicts, for each subscriber, the probability they cancel in
the next 30 days, explains why, and runs that prediction as a monitored production service.

**2. What data did you use?**
The KKBox churn dataset (WSDM Cup 2018, from Kaggle). KKBox is a music-streaming service in
Taiwan. Three files, 31.3 GB in total: members (6.77M people), transactions (payments and
plan changes), and daily listening logs (392M rows). The data is anonymised: the subscriber
ID is a 44-character hash.

**3. How exactly do you define churn?**
A subscriber "churns" if they have an explicit **cancellation** event (KKBox's `is_cancel`
flag) in the 30 days after a cutoff date. Anyone who had already cancelled before the cutoff
is excluded, because they can't churn twice. *Be ready for:* "Kaggle defines churn
differently." Yes. The competition counts anyone who doesn't renew within 30 days of expiry,
which includes quiet lapses. Ours is stricter, and that's one reason our churn rate is only
about 1.5%.

**4. What cleaning did you do?**
- **2,656,043 orphan rows dropped:** transactions for 432,623 subscribers who aren't in the
  members file (18.3%). Dropped rather than invented, because inventing a signup date would
  make long-time customers look brand new, and "new" reads as risky to the model.
- **ID column widened from 32 to 64 characters:** real IDs are 44 characters. SQLite ignored the
  limit silently; Postgres would have rejected every row.
- **Negative listening durations clipped to zero:** a known logging artefact.
- **Prices normalised to a monthly rate:** plans run 7 to 410 days, so raw prices aren't
  comparable.
- **Session history bounded to Oct 2016 – Feb 2017 (38.2M of 392M rows):** a 30-day feature
  window never needs older history, and loading all of it took 7+ hours.

**5. Why store events instead of one row per subscriber?**
Because a model is trained on the past. To train on "1 Nov 2016" we need to know what each
subscriber looked like *on that date*, not today. A table of timestamped events can answer
that for any date; a table of running totals can't, since the totals already include the
future.

**6. Why Postgres, and why is there also SQLite?**
Postgres is a real database server, and it's what the full stack runs on. SQLite is a single
file, so the project also runs on a laptop with no server at all. The code is written once
(SQLAlchemy), and one setting switches between them. CI trains on both, because SQLite is
forgiving in ways Postgres isn't: the 32-character ID bug passed silently on SQLite.

---

## Features and the model

**7. What is data leakage, and how did you prevent it?**
Leakage is when the model sees information during training that it wouldn't have when making
a real prediction. It then looks brilliant in testing and fails in production. We prevent it
with **point-in-time features**: for a cutoff date T, features use only the 30 days *before*
T, and the label only looks at the 30 days *after* T. The two windows never overlap.

**8. What are the model's inputs?**
Nine: tenure, plan type, monthly fee, sessions in the last 30 days, days since last activity,
support tickets, payment failures, discounts used, and auto-renew on/off. From these the
pipeline derives ratios such as `recency_ratio` (days inactive ÷ tenure), `fee_per_session`
and `discount_dependency`.

**9. Two inputs do nothing. Why keep them?**
Support tickets and payment failures are constant in KKBox (there's no customer-service data,
and a transaction is only recorded when money actually moved), so the model learns nothing
from them. They stay because the API's request format requires them, and the dashboard says
so openly rather than pretending they matter.

**10. Why does "auto-renew on" *raise* the risk? Isn't that backwards?**
It's what the data says. On KKBox, auto-renew is common on the cheap short plans, and those
churn the most. The model learned a real association in this data, even if it isn't a cause.
It's the single most important feature (28% of importance). The dashboard's explanations
state the actual input value, so they never claim the opposite of what you entered.

**11. Why gradient boosting?**
It's strong on tabular data with mixed numeric and categorical inputs, it handles non-linear
effects without manual feature engineering, and it works with **SHAP's exact tree
explainer**, so every prediction can be explained exactly. 300 shallow trees (depth 3) with
learning rate 0.05 keeps it from overfitting.

**12. How did you split the data, and why not randomly?**
By time: train on cutoffs in Nov and Dec 2016, validate on 31 Dec 2016, test on 29 Jan 2017.
A random split mixes future rows into training, so the test score measures memory rather
than prediction. Time-based splitting is how the model will actually be used.

---

## Metrics (be very comfortable here)

**13. Why not accuracy?**
Only about 1.5% of subscribers churn, so a "model" that says *nobody churns* is 98.5%
accurate and completely useless. Our model's accuracy is 95.6%, *lower* than that useless
baseline, because it deliberately flags people. Accuracy rewards ignoring the rare class.

**14. Why PR-AUC and not ROC-AUC?**
ROC-AUC barely notices how rare the positive class is; PR-AUC is dominated by it. We saw this
directly: moving from synthetic data (about 20% churn) to real KKBox (1.5%), **ROC-AUC went
up from 0.674 to 0.835, while PR-AUC fell from 0.345 to 0.073.** If the promotion gate had
used ROC-AUC, it would have reported an improvement when the model had become much harder to
act on. That's why the gate uses PR-AUC.

**15. PR-AUC of 0.073 sounds terrible. Is the model any good?**
Compare it with the baseline, not with 1.0. A random model's PR-AUC equals the churn rate,
0.015, so 0.073 is about **4.8× better than guessing**. At the 0.05 threshold it catches 16.8%
of churners (recall), and 7.6% of the people it flags really churn (precision), about 5×
the base rate. Honestly, it's a modest model. The data has no support or payment-failure
signal, and a stricter churn definition gives fewer positives to learn from. The engineering
around the model is the main contribution.

**16. How did you choose the decision threshold?**
The served threshold, 0.05, maximised F1 on the validation set, and it sits at the bottom of
the search range, so F1 would have liked it even lower. Separately, a **cost audit** prices
each mistake using placeholder costs (missed churner £240, offer £20, offer works 30% of the
time). Under those costs, the cheapest threshold is **0.43**, saving about £16.5k on the test
set, because at a 1.5% churn rate almost no one is risky enough to be worth an offer. The
audit reports this rather than silently changing the threshold, because the right answer
depends on real business costs we don't have.

**17. What is calibration, and is your model calibrated?**
Calibrated means a predicted 5% really happens about 5% of the time. That matters when the
probability drives a money decision. Our expected calibration error is **0.004**, which is
very good. The model slightly under-predicts overall (average predicted 1.16% vs actual 1.53%).

**18. Is the model fair?**
The audit checks performance across plan types, and it **fails**: on the `standard` plan
(only 949 subscribers in the test set) the model is close to random (ROC-AUC 0.54). The
pipeline flags this as "needs attention", and there's an alert rule for it. I report it
rather than hide it. The likely cause is too few standard-plan subscribers to learn from.

---

## MLOps: the part this project is really about

**19. What makes this MLOps rather than just machine learning?**
Machine learning is training a good model once. MLOps is everything that keeps it good in
production: versioned data and parameters, a registry that records which model is live,
a gate that stops worse models shipping, serving with explanations, monitoring for drift,
alerting, automated retraining, and CI/CD that tests all of it on every change.

**20. What does MLflow do here?**
It records every training run (parameters, metrics, the model file) and holds the model
registry. The live model is whichever version has the `@champion` alias, a candidate is
`@challenger`, and switching models means moving an alias, not copying files around.

**21. How does promotion work?**
After training, the new model is compared with the current champion on the **same test data**,
using PR-AUC. It's only promoted if it's better by at least **0.005**. The margin stops two
nearly identical models from swapping back and forth on noise. CI proves it: training an
identical challenger must be *rejected*.

**22. What is shadow scoring?**
The challenger scores every live request alongside the champion, but only the champion's
answer is returned. We measure how often they agree and how many more or fewer people each
would flag, so we learn how the challenger behaves on real traffic before it can affect
anyone.

**23. What is drift, and how do you detect it?**
Drift is when live data stops looking like training data, for example people listening less
after a price change. The model was never trained on that world, so it may quietly get worse.
We use **PSI** (Population Stability Index): split a feature into 10 bins using the training
data, then compare what fraction of live data falls in each bin. PSI under 0.10 is stable,
0.10–0.25 is moderate, above 0.25 is significant.

**24. Why check drift *before* retraining?**
Retraining writes a new baseline profile from the latest data. Checking drift afterwards would
compare the data with itself and always say "stable", so it would detect nothing, ever.

**25. What does Prefect do?**
It runs the pipeline as a flow of tasks (ingest → drift → train and gate → report), with
retries and a run history, and it can schedule it nightly. There's also a monitoring-only
flow that checks drift without retraining.

**26. What does "idempotent" mean, and why does it matter?**
Running something twice gives the same result as running it once. If the nightly job fails
halfway and gets re-run, it mustn't double every subscriber's history. CI runs the pipeline
twice and checks that the second run changes nothing and rejects the identical model.

**27. What is DVC for?**
Versioning data and pipeline stages the way Git versions code. `dvc.yaml` declares each stage,
its inputs and its outputs; `params.yaml` holds every setting. `dvc repro` re-runs only the
stages whose inputs changed, so any result can be traced back to exact data and parameters.

**28. How does monitoring work end to end?**
The API and the stream scorer expose metrics. **Prometheus** pulls them every few seconds and
evaluates 15 alert rules (model not loaded, API down, significant drift, pipeline stale,
fairness disparity, scorer down, …). Firing alerts go to **Alertmanager**, which groups them
and routes them by severity. **Grafana** draws the dashboard.

**29. What does the streaming part do?**
Events arrive on a Kafka topic (Redpanda locally). The scorer reads each one, predicts, and
writes the result to an output topic. A malformed message goes to a **dead-letter topic**
instead of crashing the consumer or disappearing. CI sends deliberately broken "poison"
messages to prove this.

**30. Why Docker, and why is the model mounted into the container?**
Docker makes the service run identically everywhere. The image contains a model trained at
build time on synthetic data, because the real model file isn't in Git. For the real stack,
the model the registry promoted is mounted in read-only, so the container serves exactly the
champion.

**31. What does your CI/CD do?**
Seven GitHub Actions jobs on every push: lint and 400+ tests; train and gate on SQLite; train
and gate on Postgres, plus run the demo queries; the full pipeline (idempotency, challenger
rejection, injected drift detected); the streaming loop with poison messages; build and
smoke-test the Docker image; and on `main`, **publish the image to GitHub's container
registry**, tagged with the commit so any deployment can be traced back and rolled back.

**32. Is it deployed?**
It runs as a full stack of 8 services on Docker Compose, and there are Kubernetes manifests
for the API and scorer. It isn't on a public cloud: a 15 GB warehouse plus MLflow, Kafka and
Grafana doesn't fit a free tier. That's a deliberate scope decision, stated openly.

---

## "Tell us about a problem you found." Three real ones

**36. A bug found by cross-checking two databases.**
After moving the warehouse to Postgres, I built the same training snapshot from SQLite and
from Postgres and compared them row by row. They differed by **one subscriber** out of 1,104.
The cause: SQLite stores dates as *text*, so `'2016-12-01' < '2016-12-01 00:00:00'` is
**true** (a shorter string sorts first). Everyone who signed up *on* the cutoff day was
counted as signing up before it. Postgres compares real dates and got it right. No future
data reached the model (a later step dropped those people), but they took slots in the
sample. Fixed by comparing the date column against a real date, with a regression test that
failed before the fix and passes after. *Lesson:* SQLite is forgiving in ways that hide bugs,
which is why CI now runs on both databases.

**37. False alerts found by actually looking at the dashboard.**
Every automated check passed, but the Grafana screenshot showed two values in every panel
(`LOADED | NO MODEL`). The streaming scorer shares the API's metric registry but never
updates the API's state gauges, so it exported their defaults, and the alert rules matched
those zeros: *PipelineStale* was firing on a healthy system. Fixed by scoping every
API-state query to `job="subscriber-api"`, with a test that fails if any query isn't scoped.
The same look found two dashboard sections drawn on top of each other, and a "REJECTED"
promotion that never happened (a gauge's default 0 means "rejected"; it now reports "not run").
*Lesson:* a passing check isn't the same as a correct screen.

**38. Why a real database mattered, in one number.**
The same feature query took **506 seconds on SQLite and 6.8 seconds on Postgres, about 74×
faster**, on identical data with identical indexes. Postgres runs aggregations in parallel
and plans joins far better; SQLite is a single file with a simpler planner.

---

## Limitations, and the question you should expect

**33. What are the biggest limitations?**
1. The model is modest (PR-AUC 0.073) because the data lacks strong signals.
2. The fairness audit fails for the `standard` plan.
3. The cost numbers are placeholders, so the cost-optimal threshold is illustrative.
4. It isn't deployed to a public cloud.
5. Our churn label is stricter than the competition's.

**34. What would you do next?**
Add richer behavioural features from the full 392M-row logs (listening trends, skips), try
the competition's churn definition, collect real offer costs so the threshold can be set by
money rather than F1, and deploy to a managed Kubernetes cluster.

**35. How did you build this? Did you use AI tools?**
Answer this truthfully. Many colleges have a policy on AI assistance, so check yours before
the viva. If you used an AI coding assistant, say so plainly and explain what *you* did:
chose the problem and data, made or approved the design decisions, ran and verified
everything on your machine, and understood each part well enough to answer every question on
this sheet. Being open about the tools and clearly understanding the system is far stronger
than a claim that falls apart under one follow-up question.
