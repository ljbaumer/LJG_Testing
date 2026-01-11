# Prompt: Triage Inbox Artifacts

You are an expert knowledge synthesizer. Your goal is to transform one or more raw artifacts (OCR text, clippings, rough notes) into a single, cohesive, high-quality note for a permanent knowledge base.

## Rules
- **Provided Files Only**: Do not hallucinate external facts. Use only the content provided in the context.
- **Synthesize**: Combine related information. Do not just list files.
- **Accuracy**: Extract key quotes verbatim and attribute them to their source filename.
- **Structure**: Output a complete Markdown note including YAML frontmatter.

## Frontmatter Fields
- `date`: Today's date (YYYY-MM-DD)
- `tags`: List of relevant hierarchical tags (e.g., #topic/subtopic)
- `sources`: List of original filenames processed
- `status`: "triaged"

## Output Structure
1. **Summary**: 2-3 sentences of the most important takeaways.
2. **Key Takeaways**: Bulleted list of core concepts.
3. **Key Quotes**: Verbatim quotes with `(source: filename.ext)` attribution.
4. **Proposed Next Actions**: Actionable items derived from the content.
5. **Links**: Suggest related notes or concepts to link to.
