# Deploying Market Dashboard to Streamlit Cloud

## Quick Start (5 minutes)

### Step 1: Push to GitHub
```bash
# If not already a git repo
git init
git add .
git commit -m "Initial commit"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/market-dashboard.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/market-dashboard`
5. Set **Main file path**: `dashboard/app.py`
6. Click **"Deploy"**

### Step 3: Add Your API Keys
1. After deployment, click **"Settings"** (gear icon)
2. Go to **"Secrets"** section
3. Paste your secrets in TOML format:

```toml
FRED_API_KEY = "your_actual_fred_key_here"
NASDAQ_DATA_LINK_KEY = "your_actual_nasdaq_key_here"
SETTINGS_PAGE_PASSWORD = "your_settings_password"
```

4. Click **"Save"**
5. Your app will restart with the new keys

---

## Getting API Keys

### FRED API Key (Required)
1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a free account
3. Request an API key (instant approval)

### Nasdaq Data Link Key (Optional but Recommended)
1. Go to https://data.nasdaq.com
2. Create a free account
3. Go to Account Settings → API Key
4. Copy your key

**Note:** Without Nasdaq key, COT data will still work using CFTC direct downloads (slightly slower).

---

## Sharing Your Dashboard

Once deployed, you'll get a URL like:
```
https://YOUR_APP_NAME.streamlit.app
```

Share this link with friends! They can view the dashboard without needing their own API keys.

---

## Updating Your Dashboard

Any push to your GitHub repo will auto-deploy:
```bash
git add .
git commit -m "Update dashboard"
git push
```

Streamlit Cloud will automatically rebuild and deploy within ~1 minute.

---

## Troubleshooting

### "FRED_API_KEY not found"
- Check Secrets are set in Streamlit Cloud settings
- Ensure format is `FRED_API_KEY = "your_key"` (with quotes)

### App crashes on startup
- Check the logs in Streamlit Cloud dashboard
- Common issue: missing dependency in `requirements.txt`

### Data not loading
- Some data sources may timeout on first load
- Refresh the page to retry
- Check if your FRED API key is valid

### COT data shows "No data available"
- This is normal if markets were closed
- COT data updates weekly (Fridays)
- Try again after market hours

---

## Local Development

To run locally with secrets:
```bash
# Copy the example secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Edit with your actual keys
nano .streamlit/secrets.toml

# Run the app
streamlit run dashboard/app.py
```

---

## Cost

**Free tier includes:**
- 1 private app
- Unlimited public apps
- Community support

This is typically more than enough for personal dashboards!
