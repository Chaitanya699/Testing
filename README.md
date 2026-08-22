# AI Sentiment Analyzer — Full Roadmap (Train → Interface → Host Live)

Project: DL (LSTM) model trained from scratch on IMDB movie reviews,
served through a Flask API + simple web UI, with an optional GenAI
(Groq LLM) explanation layer. Hosted live on Render (free tier).

Covers: AI (the whole app) + ML/DL (LSTM trained from scratch) + GenAI (optional Groq explanation).

---

## Files in this project
```
ai-sentiment-app/
├── train.py            <- trains the DL model (run once, locally)
├── app.py               <- Flask backend, serves predictions + UI
├── templates/
│   └── index.html        <- web interface
├── requirements.txt      <- dependencies
├── Procfile               <- tells Render how to run the app
└── README.md               <- this file
```

## STEP 1 — Local setup
```bash
cd ai-sentiment-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## STEP 2 — Train the model FROM SCRATCH
```bash
python train.py
```
- Downloads IMDB dataset (25k reviews) automatically via Keras.
- Builds an Embedding → LSTM → Dense architecture.
- Trains for 3 epochs (~5-10 min on CPU, faster on GPU).
- Saves `sentiment_model.h5` and `tokenizer.pkl` in this folder.
- You'll see accuracy printed at the end (~85-88% typical for this setup).

Want better accuracy? Open `train.py` and bump `EPOCHS = 3` to `5` or `10`.

## STEP 3 — Run it locally to test
```bash
python app.py
```
Open `http://localhost:5000` in your browser. Type a review, hit
"Analyze Sentiment" — you'll see Positive/Negative + confidence %.

## STEP 4 — (Optional) Enable the GenAI explanation layer
1. Get a free API key from https://console.groq.com
2. Locally: `export GROQ_API_KEY=your_key_here` (Windows: `set GROQ_API_KEY=...`)
3. Restart `python app.py` — now each prediction also shows a one-line
   AI-generated explanation.
4. Skip this step entirely if you just want the core ML/DL app — it
   works fine without it.

## STEP 5 — Push to GitHub
```bash
git init
git add .
git commit -m "AI sentiment analyzer - DL model + Flask interface"
gh repo create ai-sentiment-app --public --source=. --push
# or manually create a repo on github.com and:
# git remote add origin https://github.com/<your-username>/ai-sentiment-app.git
# git branch -M main
# git push -u origin main
```
**Important:** `sentiment_model.h5` and `tokenizer.pkl` must be pushed
too (don't gitignore them) — Render needs these files to serve
predictions, since it doesn't run `train.py` for you.

## STEP 6 — Host live on Render (free tier)
Why Render and not Firebase: Firebase Hosting only serves static
files (HTML/CSS/JS) or lightweight Cloud Functions — it can't run a
Flask + TensorFlow backend directly. Render (like Railway, which
you've already used before) runs full Python web services for free,
which is what this app needs.

1. Go to https://render.com → Sign up / log in with GitHub.
2. Click **New +** → **Web Service**.
3. Connect your GitHub repo (`ai-sentiment-app`).
4. Fill in:
   - **Name:** ai-sentiment-app
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Under **Environment Variables** (optional):
   - `GROQ_API_KEY` = your key (only if you want Step 4's feature live)
6. Click **Create Web Service**. Render builds and deploys — takes
   ~5-10 min because TensorFlow is a big install.
7. You'll get a live URL like:
   `https://ai-sentiment-app.onrender.com`

That's it — model trained from scratch, interface built, hosted live.

## STEP 7 — (If you specifically want Firebase)
Firebase can still work, but only for the **frontend**:
- Deploy `templates/index.html` as a static Firebase Hosting site.
- Point its `fetch("/predict")` calls to your Render backend URL
  instead of a relative path (e.g. `https://ai-sentiment-app.onrender.com/predict`).
- This is more setup for no real benefit here — Render alone is
  simpler since it serves both backend AND frontend together.

## Notes / troubleshooting
- Free Render services sleep after 15 min idle — first request after
  sleep takes ~30-50 sec to "wake up". Normal on free tier.
- If build fails on TensorFlow size limits, switch `tensorflow` to
  `tensorflow-cpu` in `requirements.txt` (smaller install).
- To improve the model later: try Bidirectional LSTM, GRU, or swap in
  a pretrained embedding (GloVe) instead of training embeddings from
  scratch.
