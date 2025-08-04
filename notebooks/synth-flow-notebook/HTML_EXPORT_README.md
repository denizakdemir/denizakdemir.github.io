# HTML Export Instructions

This notebook has been set up to export to HTML with collapsible code cells and hidden package installation outputs.

## Method 1: Using the Custom HTML/CSS (Already in Notebook)

The first cell of the notebook contains HTML/CSS/JavaScript that will:
- Add "Show/Hide Code" buttons to all code cells
- Hide package installation outputs automatically
- Code cells are hidden by default, outputs are visible

Simply export the notebook as-is:
```bash
jupyter nbconvert --to html synthetic-data-blog-notebook.ipynb
```

## Method 2: Using Custom Template (Recommended)

Use the provided conversion script:
```bash
python convert_to_html.py synthetic-data-blog-notebook.ipynb
```

This will:
1. Tag cells with package installation to hide their output
2. Apply a custom template with collapsible code cells
3. Generate an HTML file with clean, professional formatting

## Method 3: Manual Export with Tags

1. In Jupyter, add tags to cells you want to hide:
   - View → Cell Toolbar → Tags
   - Add "hide-output" tag to package installation cells

2. Export with tag removal:
```bash
jupyter nbconvert --to html \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_all_outputs_tags=hide-output \
  synthetic-data-blog-notebook.ipynb
```

## Features in the Exported HTML

- **Collapsible Code**: All code cells have a "Show/Hide Code" button
- **Hidden Installation Output**: Package installation outputs are automatically hidden
- **Clean Presentation**: Focus on results and visualizations
- **Interactive**: Readers can choose to view code when interested

## Customization

Edit `custom_template.tpl` to change:
- Button styling (color, size, position)
- Default visibility (show/hide code by default)
- Which cells to hide (based on content patterns)