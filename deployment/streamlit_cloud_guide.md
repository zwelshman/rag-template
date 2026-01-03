# Deploy to Streamlit Cloud

This guide will walk you through deploying your Streamlit RAG Kit to Streamlit Cloud.

## Prerequisites

- A GitHub account
- Your repository pushed to GitHub
- An Anthropic API key

## Step-by-Step Deployment

### 1. Push Your Code to GitHub

```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### 2. Sign Up for Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Grant Streamlit access to your repositories

### 3. Deploy Your App

1. Click "New app" in Streamlit Cloud
2. Select your repository
3. Choose the branch (usually `main`)
4. Set the main file path: `app.py`
5. Click "Deploy"

### 4. Configure Secrets

1. In your app's dashboard, click "Settings" → "Secrets"
2. Add your API key:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-api-key-here"
```

3. Save the secrets

### 5. Verify Deployment

- Your app should be live at `https://[your-app-name].streamlit.app`
- Test document upload and question answering
- Check logs for any errors

## Configuration Tips

### Custom Domain

1. Go to Settings → General
2. Add your custom domain
3. Update DNS records as instructed

### Resource Limits

Streamlit Cloud Free Tier:
- 1 GB RAM
- 1 CPU core
- Storage is ephemeral

For production apps with higher traffic:
- Consider Streamlit Cloud Teams or Enterprise
- Or deploy to your own infrastructure

### Optimization for Cloud

Add to your `requirements.txt`:
```txt
# Keep dependencies minimal
streamlit>=1.28.0
anthropic>=0.7.0
chromadb>=0.4.18
# ... other essential packages
```

## Troubleshooting

### App Not Starting

1. Check deployment logs in Streamlit Cloud
2. Verify `requirements.txt` is correct
3. Ensure API keys are set in Secrets

### Out of Memory

- Reduce chunk size in settings
- Limit number of documents
- Consider upgrading to Teams plan

### Slow Performance

- Enable caching with `@st.cache_data`
- Use smaller embedding models
- Reduce number of retrieved chunks

## Monitoring

- View app usage in Streamlit Cloud dashboard
- Check logs for errors
- Monitor API usage in Anthropic console

## Next Steps

- Set up custom domain
- Configure analytics
- Add user authentication
- Scale with Streamlit Teams

For more information, visit [docs.streamlit.io](https://docs.streamlit.io)
