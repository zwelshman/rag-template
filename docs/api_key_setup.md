# API Key Setup

This guide explains how to set up API keys for different environments.

## Anthropic API Key (Required)

### Get Your API Key

1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign in or create an account
3. Navigate to "API Keys"
4. Click "Create Key"
5. Copy your API key (starts with `sk-ant-`)

### Local Development

Create `.streamlit/secrets.toml` in your project root:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-api-key-here"
```

**Important**: Add `.streamlit/secrets.toml` to `.gitignore` to avoid committing secrets!

### Environment Variables

Alternatively, set an environment variable:

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="sk-ant-your-api-key-here"

# Windows
set ANTHROPIC_API_KEY=sk-ant-your-api-key-here
```

### Streamlit Cloud

1. Go to your app settings in Streamlit Cloud
2. Click "Secrets"
3. Add:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-api-key-here"
```

### Docker

Create a `.env` file:

```env
ANTHROPIC_API_KEY=sk-ant-your-api-key-here
```

Then use with docker-compose:

```bash
docker-compose --env-file .env up
```

## Optional: OpenAI API Key

If you want to use OpenAI models:

```toml
OPENAI_API_KEY = "sk-your-openai-api-key"
```

Get your key from [OpenAI Platform](https://platform.openai.com/).

## Security Best Practices

1. **Never commit secrets** to version control
2. **Use environment-specific secrets** (dev, staging, prod)
3. **Rotate keys regularly**
4. **Monitor API usage** in provider dashboards
5. **Set spending limits** in provider consoles
6. **Use read-only keys** when possible

## Troubleshooting

### "Invalid API key" Error

- Check key format (should start with `sk-ant-`)
- Verify no extra spaces or quotes
- Ensure key is active in provider console

### "API key not found" Error

- Check file location: `.streamlit/secrets.toml`
- Verify TOML syntax
- Restart Streamlit app after changing secrets

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
    key = st.secrets["ANTHROPIC_API_KEY"]
    print("✓ API key found in secrets")
except:
    print("✗ API key not found in secrets")

# Check environment
env_key = os.getenv("ANTHROPIC_API_KEY")
if env_key:
    print("✓ API key found in environment")
else:
    print("✗ API key not found in environment")
```
