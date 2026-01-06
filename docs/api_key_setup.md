# API Key Setup

This guide explains how to set up API keys for different environments.

## Hugging Face API Key (Required)

### Get Your API Key

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Sign in or create an account
3. Navigate to "Access Tokens"
4. Click "New token"
5. Copy your API key (starts with `hf_`)

### Local Development

Create `.streamlit/secrets.toml` in your project root:

```toml
HF_API_KEY = "hf_your-huggingface-api-key-here"
```

**Important**: Add `.streamlit/secrets.toml` to `.gitignore` to avoid committing secrets!

### Environment Variables

Alternatively, set an environment variable:

```bash
# Linux/Mac
export HF_API_KEY="hf_your-huggingface-api-key-here"

# Windows
set HF_API_KEY=hf_your-huggingface-api-key-here
```

### Streamlit Cloud

1. Go to your app settings in Streamlit Cloud
2. Click "Secrets"
3. Add:

```toml
HF_API_KEY = "hf_your-huggingface-api-key-here"
```

### Docker

Create a `.env` file:

```env
HF_API_KEY=hf_your-huggingface-api-key-here
```

Then use with docker-compose:

```bash
docker-compose --env-file .env up
```

## Supported Models

The default model is `meta-llama/Meta-Llama-3.1-8B-Instruct`. The provider includes automatic fallback support for deprecated models:

| Model | Description |
|-------|-------------|
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | Fast, good quality (default) |
| `Qwen/Qwen2.5-7B-Instruct` | Good alternative |
| `microsoft/Phi-3-mini-4k-instruct` | Compact, efficient |

## Optional: OpenAI API Key

If you want to use OpenAI models instead:

```toml
OPENAI_API_KEY = "sk-your-openai-api-key"
```

Get your key from [OpenAI Platform](https://platform.openai.com/).

## Optional: Anthropic API Key

If you want to use Anthropic Claude models:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-anthropic-api-key"
```

Get your key from [Anthropic Console](https://console.anthropic.com/).

## Security Best Practices

1. **Never commit secrets** to version control
2. **Use environment-specific secrets** (dev, staging, prod)
3. **Rotate keys regularly**
4. **Monitor API usage** in provider dashboards
5. **Set spending limits** in provider consoles
6. **Use read-only keys** when possible

## Troubleshooting

### "Invalid API key" Error

- Check key format (should start with `hf_` for Hugging Face)
- Verify no extra spaces or quotes
- Ensure key is active in provider console

### "API key not found" Error

- Check file location: `.streamlit/secrets.toml`
- Verify TOML syntax
- Restart Streamlit app after changing secrets

### "Model deprecated" Error

The application has automatic fallback support. If you see this error, the system will automatically try alternative models. Consider updating your configuration to use one of the supported models listed above.

### "Rate limit exceeded" Error

- Check usage limits in provider console
- Implement rate limiting in your app
- Consider upgrading your API plan

## Testing Your Setup

Run this Python code to test:

```python
import os
import streamlit as st

# Check secrets
try:
    key = st.secrets["HF_API_KEY"]
    print("✓ API key found in secrets")
except:
    print("✗ API key not found in secrets")

# Check environment
env_key = os.getenv("HF_API_KEY")
if env_key:
    print("✓ API key found in environment")
else:
    print("✗ API key not found in environment")
```
