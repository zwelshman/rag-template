# Hugging Face Spaces Deployment Guide

This guide provides step-by-step instructions for deploying the RAG Template to Hugging Face Spaces.

## Prerequisites

- A Hugging Face account ([sign up here](https://huggingface.co/join))
- An Anthropic API key ([get one here](https://console.anthropic.com/))
- Git installed on your machine (optional, for Git-based deployment)

## Deployment Methods

### Method 1: Direct GitHub Integration (Recommended)

This is the easiest method if your code is already on GitHub.

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for HF Spaces deployment"
   git push origin main
   ```

2. **Create a new Space on Hugging Face**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Fill in the details:
     - **Owner**: Your username or organization
     - **Space name**: Choose a name (e.g., `rag-template-demo`)
     - **License**: MIT
     - **Select the Space SDK**: Choose "Streamlit"
     - **Space hardware**: CPU basic (free tier works fine)
     - **Visibility**: Public or Private

3. **Link to your GitHub repository**
   - In the Space creation form, look for "Link to a GitHub repository"
   - Authorize Hugging Face to access your GitHub
   - Select your repository

4. **Add the API key as a Secret**
   - Once the Space is created, go to "Settings" tab
   - Scroll to "Repository secrets"
   - Click "Add a secret"
   - Name: `ANTHROPIC_API_KEY`
   - Value: Paste your Anthropic API key
   - Click "Save"

5. **Your app is live!**
   - The Space will automatically build and deploy
   - Visit `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

### Method 2: Direct Git Push to HF Spaces

If you prefer to push directly to Hugging Face (without GitHub):

1. **Create a new Space**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Streamlit" as SDK
   - Click "Create Space"

2. **Clone the Space repository**
   ```bash
   # HF provides the exact command in your Space's "Files and versions" tab
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME
   ```

3. **Copy your project files**
   ```bash
   # From your rag-template directory, copy all files
   cp -r ../rag-template/* .
   cp -r ../rag-template/.streamlit .
   cp ../rag-template/.gitignore .
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Initial deployment of RAG Template"
   git push
   ```

5. **Add the API key secret** (same as Method 1, step 4)

### Method 3: Manual File Upload

For quick testing without Git:

1. **Create a new Space** (same as Method 2, step 1)

2. **Upload files via web interface**
   - In your Space, click "Files and versions"
   - Click "Add file" → "Upload files"
   - Drag and drop all your project files:
     - `app.py`
     - `requirements.txt`
     - `README.md`
     - `.streamlit/config.toml`
     - `utils/` folder (all Python files)
   - Click "Commit changes to main"

3. **Add the API key secret** (same as Method 1, step 4)

## Configuration Details

### Files Configured for HF Spaces

1. **README.md** - Contains YAML front matter that HF Spaces reads:
   ```yaml
   ---
   title: RAG Template
   emoji: 📚
   colorFrom: blue
   colorTo: purple
   sdk: streamlit
   sdk_version: "1.28.0"
   app_file: app.py
   pinned: false
   license: mit
   ---
   ```

2. **.streamlit/config.toml** - Streamlit configuration:
   - Port 7860 (HF Spaces default)
   - Headless mode enabled
   - CORS and XSRF disabled for HF environment

3. **requirements.txt** - All Python dependencies are automatically installed

### How API Key is Loaded

The app automatically loads the API key from Hugging Face Secrets:

1. First tries: `st.secrets["ANTHROPIC_API_KEY"]` (HF Spaces secrets)
2. Falls back to: `os.environ.get("ANTHROPIC_API_KEY")`

No code changes needed - the app is already configured!

## Updating Your Space

### If using GitHub integration:
```bash
# Make changes locally
git add .
git commit -m "Update app"
git push origin main
# HF Spaces will automatically rebuild
```

### If pushing directly to HF:
```bash
# Make changes locally
git add .
git commit -m "Update app"
git push
```

## Troubleshooting

### App won't start

1. **Check build logs**
   - Go to your Space → "Logs" tab
   - Look for errors during pip install or app startup

2. **Common issues**:
   - Missing API key secret → Add it in Settings
   - Wrong SDK version → Check README.md front matter
   - Missing dependencies → Verify requirements.txt

### API key not working

1. **Verify secret name is exactly**: `ANTHROPIC_API_KEY`
2. **Check the key is valid** at https://console.anthropic.com/
3. **Restart the Space**: Settings → "Factory reboot"

### Out of memory errors

1. **Reduce chunk processing**:
   - Use smaller chunk sizes
   - Process fewer documents at once
   - Consider upgrading to a larger hardware tier

2. **Upgrade hardware**:
   - Go to Settings → "Space hardware"
   - Choose a larger tier (may require payment)

## Performance Optimization for HF Spaces

### Free Tier (CPU Basic)
- Works well for demos and testing
- Handles 5-10 documents efficiently
- Response time: 2-5 seconds per query

### Recommended Settings for Free Tier:
- Chunk size: 500-1000 tokens
- Number of results: 3-5
- Max documents: 10-20 pages

### Paid Tiers
For production use with many documents:
- CPU Upgrade ($0.03/hour): Better for 50+ documents
- GPU T4 ($0.60/hour): Best for large document sets

## Security Best Practices

1. **Never commit API keys to Git**
   - Always use HF Secrets
   - Keep .env files in .gitignore

2. **Use private Spaces** for sensitive data
   - In Space creation, choose "Private"
   - Only you and selected users can access

3. **Rotate API keys** regularly
   - Update in HF Secrets settings
   - No code changes needed

## Monitoring Your Space

### View Usage Stats
- Go to your Space → "Analytics" tab
- See visitor counts, API usage

### Check Logs
- "Logs" tab shows real-time application logs
- Useful for debugging errors

## Next Steps

After deployment:
1. Test the app thoroughly
2. Share the Space URL with users
3. Monitor usage and costs
4. Consider adding custom domain (HF Pro feature)

## Support

- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Streamlit Docs**: https://docs.streamlit.io/
- **Anthropic API**: https://docs.anthropic.com/

## Example Deployed Spaces

Here are examples of similar Streamlit apps on HF Spaces:
- Search for "RAG" on https://huggingface.co/spaces
- Look at Streamlit demos for inspiration
