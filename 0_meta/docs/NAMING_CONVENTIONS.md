# Naming Conventions

Maintain a clean vault by following these naming patterns.

## General Principles
- Use **kebab-case** (lowercase with hyphens) for all filenames.
- Avoid special characters except hyphens.
- Use dates in `YYYY-MM-DD` format.

## Folder Specifics

### 1_inbox
- Prefix with date if relevant: `2024-01-20-initial-thoughts.md`
- OCR sidecars: `image-name.jpg` -> `image-name.txt`

### 4_notes (Permanent Notes)
- Descriptive and atomic: `large-language-model-quantization.md`
- Focus on the concept, not the source.

### 5_tasks
- `todos.md` for high-level lists.
- Project-specific tasks: `project-name-tasks.md`

### 6_crm
- `last-name-first-name.md`

### 7_templates
- Use descriptive titles for easy selection in Obsidian.

## Examples
- ✅ `2024-03-15-meeting-with-jane.md`
- ❌ `Meeting With Jane 3/15.md`
- ✅ `neural-network-hyperparameters.md`
- ❌ `Neural (Network) Hyper-parameters!.md`
