# Prompt Templates

This directory contains pre-configured prompt templates for different use cases.

## Available Templates

### `default_system.txt`
General-purpose system prompt for balanced responses.
- **Use case**: Most queries
- **Style**: Balanced detail and brevity
- **Citations**: Yes

### `detailed_analysis.txt`
For in-depth analysis and comprehensive answers.
- **Use case**: Complex questions requiring thorough analysis
- **Style**: Detailed and structured
- **Citations**: Extensive with page numbers

### `concise_summary.txt`
For brief, to-the-point answers.
- **Use case**: Quick lookups, simple questions
- **Style**: Very concise (under 3 sentences)
- **Citations**: Brief

## Creating Custom Prompts

1. Create a new `.txt` file in this directory
2. Write your system prompt instructions
3. Reference it in your code:

```python
with open('prompts/your_prompt.txt', 'r') as f:
    custom_prompt = f.read()

pipeline.system_prompt = custom_prompt
```

## Best Practices

1. **Be specific**: Clearly define the assistant's role
2. **Set boundaries**: Specify what to do when information is missing
3. **Citation format**: Define how sources should be cited
4. **Response style**: Specify length, tone, and structure
5. **Edge cases**: Handle ambiguous or insufficient context

## Examples by Domain

### Legal/Compliance
- Request exact citations with section numbers
- Emphasize accuracy over interpretation
- Require explicit statements about missing information

### Technical Documentation
- Focus on step-by-step instructions
- Include code examples when relevant
- Cite specific documentation sections

### Research/Academic
- Request comprehensive analysis
- Require multiple source citations
- Highlight confidence levels in conclusions

### Customer Support
- Friendly, helpful tone
- Actionable answers
- Escalation paths for complex issues
