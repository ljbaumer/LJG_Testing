# AI Drafting Vault

A structured Obsidian vault designed for AI-assisted knowledge management and drafting.

## Folder Structure

- **0_meta/**: Vault configuration, scripts, and AI prompt templates.
    - `0_meta/prompts/`: System and user prompts for AI triage and drafting.
    - `0_meta/scripts/`: CLI tools for vault maintenance (OCR, cleanup).
    - `0_meta/docs/`: Documentation for the vault's processes.
- **1_inbox/**: The landing zone for all new information. Raw notes, clippings, and images.
- **2_applications/**: Project-specific content and final applications of knowledge.
- **3_drafts/**: Active writing projects and long-form compositions.
- **4_notes/**: Atomic, permanent notes (Zettelkasten style).
- **5_tasks/**: Actionable items and project management.
- **6_crm/**: Contact management and interaction logs.
- **7_templates/**: Obsidian templates for recurring note types.

## Workflow

1. **Capture (`1_inbox`)**: Input raw thoughts, URLs, or images. 
2. **OCR Sidecars**: Images placed in `1_inbox` should have a companion `.txt` file generated containing extracted text. This allows AI to "read" your images.
3. **Refine (`4_notes`)**: Move processed information into permanent notes. Link heavily.
4. **Draft (`3_drafts`)**: Use prompt templates in `0_meta/prompts` to synthesize notes into drafts.
5. **Applied (`2_applications`)**: Final outputs go here.

## Naming Conventions

- Files should be lowercase with hyphens (e.g., `ai-drafting-system.md`).
- Metadata folders start with `0_`.
- Process folders are numbered `1-7`.

## AI Integration

- Use the scripts in `0_meta/scripts` to perform batch OCR or triage.
- Leverage the prompt templates for consistent LLM interactions.
