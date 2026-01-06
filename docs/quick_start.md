# Quick Start Guide

Get up and running with Streamlit RAG Kit in 5 minutes.

## 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd streamlit-rag-kit

# Install dependencies
pip install -r requirements.txt
```

## 2. Set Up API Key

Create `.streamlit/secrets.toml`:

```toml
HF_API_KEY = "hf_your-huggingface-api-key-here"
```

Get your API key from [Hugging Face Settings](https://huggingface.co/settings/tokens).

## 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 4. Use the App

1. **Initialize Pipeline**: Click "Initialize Pipeline" in the sidebar
2. **Upload Documents**: Go to "Upload Documents" tab and upload your files
3. **Ask Questions**: Go to "Ask Questions" tab and start chatting

## 5. Configuration

Adjust settings in the sidebar:

- **Search Mode**: Hybrid (recommended), Vector, or BM25
- **Chunk Size**: Document chunk size (default: 1000)
- **Temperature**: LLM creativity (0-1, default: 0.7)
- **Max Tokens**: Response length (default: 1000)

## Next Steps

- [Customize the prompts](./customization_guide.md)
- [Deploy to Streamlit Cloud](../deployment/streamlit_cloud_guide.md)
- [Learn about authentication](./authentication_setup.md)
- [Troubleshooting](./troubleshooting.md)

## Common Issues

### "No API key found"

Make sure `.streamlit/secrets.toml` exists with your API key.

### "Module not found"

Run `pip install -r requirements.txt` to install all dependencies.

### App is slow

- Reduce chunk size
- Lower number of results
- Use BM25 search mode instead of Hybrid

## Support

For issues and questions:
- Check [Troubleshooting Guide](./troubleshooting.md)
- Open an issue on GitHub
- Check [Hugging Face documentation](https://huggingface.co/docs/huggingface_hub/)
