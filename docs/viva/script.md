# The 18-minute demo

What to click, and what to say while you click it. The words are a guide. Say them in your
own way, but keep the order, because each part sets up the next.

---

## One-time setup: TablePlus (10 minutes, do this days before)

1. Download TablePlus from **tableplus.com** → open the `.dmg` → drag TablePlus into
   Applications → open it. The free version is enough. It never expires; it only limits you
   to 2 open tabs and 2 windows.
2. Click **Create a new connection…** → choose **PostgreSQL** → **Create**.
3. Fill in exactly:

   | Field | Value |
   |---|---|
   | Name | `Subscriber Warehouse` |
   | Host | `127.0.0.1` |
   | Port | `5432` |
   | User | `subscriber` |
   | Password | `subscriber` |
   | Database | `warehouse` |

4. Click **Test**. Every field should turn green (the stack must be running: `make demo`).
   Then **Save**, then **Connect**.
5. The left sidebar lists the tables. Click `subscribers` to see rows.
6. **Running a query:** press **⌘E** to open a SQL tab, paste a block from
   `DEMO_QUERIES_POSTGRES.sql`, put the cursor inside it, and press **⌘↵** (Cmd+Return) to run
   just that query. **⇧⌘↵** runs everything; don't use it in the demo.
7. With the 2-tab limit, keep one tab showing a table and one SQL tab. Reuse the SQL tab: select
   all (⌘A), paste the next query, run.

The sidebar shows exactly six tables: the five warehouse tables plus `warehouse_summary`.

---

## The day before

```bash
make demo-prepare     # builds the image, writes the drift report (~5 min)
make demo             # start everything
make demo-check       # must say 19/19
```

Then open every page once in your own browser, so first-visit pop-ups are gone on the day:

- **MLflow** http://127.0.0.1:5050: close the "new Model Registry UI" pop-up and the
  Assistant panel on the right. Click **Model training** (top left) instead of GenAI.
- **Grafana** http://127.0.0.1:3000: Dashboards → *Subscriber Dropout — Model Health*.
  Set the time range (top right) to **Last 30 minutes**.

Then `make demo-down`.

## 30 minutes before

1. Plug in the charger. Turn on Do Not Disturb. Quit everything you don't need (Slack,
   mail, anything that pops up).
2. `make demo`, then `make demo-check`, and wait for **19/19 checks passed**. If anything
   fails, see "If something breaks" at the bottom.
3. Open these tabs **in this order**, left to right:
   1. `DEMO_QUERIES_POSTGRES.sql` in any editor, for copying
   2. TablePlus (connected to Subscriber Warehouse)
   3. MLflow: http://127.0.0.1:5050
   4. Dashboard: http://127.0.0.1:8000
   5. Grafana: http://127.0.0.1:3000
   6. Prometheus alerts: http://127.0.0.1:9090/alerts
   7. GitHub: the repo's **Actions** tab, showing the latest green run
   8. `docs/viva/architecture.md` preview, to show the diagram when you explain the pieces
4. Browser zoom around 110%, so the examiner can read from across a desk.

---

## The talk

### 0:00 – 1:00 · The problem (dashboard → Overview)

Open **http://127.0.0.1:8000**. It opens on the **Overview**: the live MLOps pipeline. The
header pill must name `gradient_boosting_classifier` (the tested model). If it says "live
replay model", press **Restore original model** first.

> "Subscription businesses lose revenue when people cancel. If you can spot who's *about* to
> cancel, you can act while they're still a customer. I built a system that predicts that
> for a real music-streaming service. But the model is only a small part of it. Most of
> the work is everything that keeps a model trustworthy in production: that's what MLOps
> means. Let me show it working before I explain it."

### 1:00 – 3:00 · The pipeline, live (Overview → Run)

1. Click **▶ Run 1 Jan 2017**. The eight stage cards light up one by one (about 30 seconds
   the first time, 10–30 seconds after that). Talk over it:
   > "It's pretending to be 1 January 2017 and doing what the nightly pipeline did that day,
   > on the real data in Postgres. New data arrives, features are built only from the past,
   > the data is validated, checked for drift, a model is trained, tested on a month it
   > never saw, and then the **gate** decides whether it goes live."
2. When it finishes, point at **Go live** and the header pill, which now says *live replay
   model · MLflow vN*:
   > "It was promoted, because there was no champion yet, and the website is now serving it.
   > Nothing restarted."
3. Click **▶ Run 31 Jan 2017**. This one is **rejected**: 0.0475 against 0.0474.
   > "It's a bit better, but not by the 0.005 margin, so the old model keeps serving. That's
   > the gate doing its job: a new model has to be *measurably* better, not luckily better."
4. Leave month 3 for later.

If the examiner asks whether it's real: the scores differ each month, the gate rejected one,
and every run appears in MLflow. See the book, section H.

### 3:00 – 5:30 · The data (TablePlus)

> "This is real data: KKBox, a music-streaming company, from a Kaggle competition. 31 GB of
> raw CSV, which I cleaned into a Postgres warehouse."

1. Run **query 1** (`warehouse_summary`).
   > "82.8 million rows: 6.8 million subscribers, 38 million listening sessions."
2. Click the `subscribers` table in the sidebar.
   > "IDs are 44-character hashes. The data is anonymised."
3. Run **query 4**, one subscriber's full history.
   > "This is why the warehouse stores **events with timestamps**, not totals. I can
   > rebuild what this person looked like on any past date. That's what prevents data
   > leakage in training, which I'll come back to."
   Mention it's instant: the tables are indexed on subscriber and time.
4. Run **query 5**, the cleaning log.
   > "The raw data wasn't clean. The biggest decision: 2.66 million transactions belonged
   > to subscribers missing from the members file. I **dropped** them instead of inventing
   > signup dates, because invented dates would make long-standing customers look new, and
   > the model reads 'new' as risky."

### 5:30 – 8:00 · Features, training, and the registry (MLflow)

> "For training I pick a cutoff date. Features only use the 30 days **before** it; the label
> is whether they cancel in the 30 days **after**. The windows never overlap, so the model
> can't see the answer. And I split by time, training on Nov–Dec 2016 and testing on
> January 2017, never randomly."

In MLflow: **Model training → Experiments → subscriber-dropout**, open the full training run
from 6 Sep 2026 (the one registered as `subscriber-dropout-classifier` v1). The Overview's
replay runs have their own experiment, `subscriber-dropout-live-replay`.

> "Every run records its parameters and metrics."

Then **Models → subscriber-dropout-classifier**.

> "The registry holds versions, and the live one carries the **@champion** alias. A new
> model only takes over if it beats the champion on **PR-AUC**, by a margin."

Then **Models → subscriber-dropout-live**: the versions the Overview just made, one per click,
with **@champion** on the one that's serving. Back on the dashboard, click **▶ Run 28 Feb
2017**. It's **promoted** (0.0568 vs 0.0489), and drift is *moderate*: last-activity days
shifted (PSI 0.153). Refresh MLflow and a new version carries **@champion**.

> "That's the whole loop: data in, a retrained model, a gate, and a deployment, with every
> step recorded."

Then press **Restore original model**. The Score tab's examples are tuned to the tested model.

Say the key finding. It's the strongest moment of the talk:

> "Why PR-AUC and not accuracy or ROC-AUC? Only 1.5% of subscribers churn, so a model
> that says 'nobody churns' is 98.5% accurate and useless. And when I moved from
> synthetic data to real data, **ROC-AUC went up from 0.67 to 0.84 while PR-AUC fell
> from 0.35 to 0.07.** ROC-AUC would have told me things got better when they got much
> harder. The gate uses PR-AUC because of that."

### 8:00 – 11:30 · Serving (dashboard)

On the **Score** tab (it's already scored the at-risk example):

> "The API returns a probability, a risk band, and the reasons. The reasons come from
> **SHAP**: the contributions add up exactly to the prediction, so the explanation can't
> disagree with the score."

1. Point at the three drivers (discounts, auto-renew, value per session).
2. Point at **What would change it**.
   > "Each of these is a real second request to the model with one field changed. Turning
   > auto-renew off takes this subscriber from 60% to under 1%."
3. Be honest when someone raises an eyebrow:
   > "Auto-renew *raising* risk looks backwards. On KKBox it goes with the cheap short
   > plans that churn most. The model learned a real association in this data, not a
   > cause."
4. Click **Healthy**, then **Borderline**, and watch it re-score.

### 11:30 – 15:00 · Monitoring (dashboard → Grafana → Prometheus)

1. Dashboard **Monitoring** tab: the drift test has already run.
   > "Drift is when live data stops looking like training data. I measure it with PSI. To
   > show the detector actually works, I deliberately shift two features, and it ranks
   > **exactly those two** at the top while the others stay under 0.5."
2. **Grafana**:
   > "Prometheus collects metrics from the API and the streaming scorer every few seconds,
   > and Grafana shows them."
   Point at: model loaded, prediction traffic, and **Drift verdict: STABLE**.
   > "That's a real check on February 2017 data, run through Postgres: every feature's PSI
   > is under 0.06, so the world didn't shift."
3. Point at **Fairness audit: DISPARITY**. Don't hide it; raise it yourself:
   > "The model's own audit fails: on the standard plan it's close to random, probably
   > because there are only 949 of those subscribers to learn from. The system reports it
   > and alerts on it rather than hiding it."
4. **Prometheus → Alerts**: FairnessDisparity is firing, for exactly that reason.

### 15:00 – 17:00 · Automation (GitHub Actions)

> "Every push runs 8 jobs. Beyond unit tests, CI proves the **system** behaves: it trains on
> SQLite *and* on Postgres; runs the pipeline twice and checks the second run changes
> nothing; checks that an identical challenger is **rejected**; injects drift and checks
> it's **detected**; sends broken messages to the streaming scorer and checks they go to a
> dead-letter queue; deploys to a real, throwaway Kubernetes cluster and checks the health
> probes; and on `main` it publishes the Docker image to GitHub's registry,
> tagged with the commit."

Click into the latest run and show the green jobs.

### 17:00 – 18:00 · Limitations and close

> "Four honest limitations. The model is modest, about 5× better than guessing, because the
> data lacks strong signals. The fairness audit fails for one plan. The cost numbers are
> placeholders. And it runs on Docker locally, not a public cloud. A 15 GB warehouse doesn't
> fit a free tier. The next steps would be richer listening features, real offer costs, and
> a managed Kubernetes deployment."

> "Thank you. Happy to go into any part."

---

## If something breaks

| Symptom | Do this |
|---|---|
| `make demo-check` shows a FAIL | Read the detail column; it says what's wrong. Most failures are a service still starting: wait 30 s and re-run. |
| The Run button fails or hangs | Press **Restore original model** and carry on. Everything else is independent of it. Say: "it replays a month on the real data; I'll show the gate in MLflow instead." |
| Score tab gives unexpected numbers | A replay model is still serving. Press **Restore original model** on the Overview. |
| Docker won't start | Open Docker Desktop by hand, wait for the whale icon to settle, re-run `make demo`. |
| TablePlus "connection refused" | Postgres isn't up: `docker compose up -d postgres`, wait 10 s, reconnect. |
| MLflow page blank | `make demo-down` then `make demo`. The MLflow UI is restarted with the stack. |
| Grafana panels say "No data" | Traffic ages out after a while. Click **Healthy / At-risk** a few times on the dashboard, or re-run `make demo` (it re-sends warm-up traffic). |
| Anything else, mid-talk | Don't debug live. Say "let me show you the screenshot of this instead" and switch to `docs/viva/screenshots/`. |
