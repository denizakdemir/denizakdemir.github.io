# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based personal blog focused on data science, machine learning, and statistical analysis. The site uses the Chirpy theme and is hosted on GitHub Pages at denizakdemir.github.io.

## Development Commands

### Repository Management
- `./tools/setup_blog.sh status` - Show repository health and missing blog posts
- `./tools/setup_blog.sh setup` - Install dependencies and setup environment  
- `./tools/setup_blog.sh clean` - Clean duplicate files and build artifacts
- `./tools/setup_blog.sh convert` - Interactive notebook to blog post conversion

### Local Development
- `./tools/run.sh` - Start Jekyll development server with live reload
- `./tools/run.sh -p` - Run in production mode
- `./tools/run.sh -H 0.0.0.0` - Bind to specific host

### Testing and Building
- `./tools/test.sh` - Build site for production and run HTML validation tests
- `bundle install` - Install Ruby dependencies
- `bundle exec jekyll build` - Build site manually

### Notebook Conversion
- `python3 tools/convert_notebook.py notebooks/YourNotebook.ipynb` - Convert single notebook
- `python3 tools/convert_notebook.py notebooks/YourNotebook.ipynb --title "Custom Title"` - With custom title

## Architecture

### Content Structure
- `_posts/` - Blog posts in markdown format with YAML front matter
- `_tabs/` - Static pages (About, Contact, etc.)
- `notebooks/` - Jupyter notebooks for technical demonstrations
- `models/` - Saved ML models (.pkl, .pth files)
- `data/` - Datasets used in examples

### Blog Post Format
Posts use Jekyll front matter with:
- `title`, `author`, `date`
- `categories: [Machine Learning, Tutorial]`
- `tags: [PyTorch, Anomaly Detection, etc.]`
- `render_with_liquid: false` for technical content

### Theme Configuration
- Uses Jekyll Chirpy theme v7.2.4
- Configuration in `_config.yml`
- Custom styling in `_sass/` and `assets/css/`
- PWA enabled with offline caching

### Technical Content
- Jupyter notebooks are converted to markdown for blog posts
- Code examples primarily use PyTorch, scikit-learn, pandas
- Focus areas: anomaly detection, transformers, rule mining, clinical data analysis
- Images automatically handled during conversion process

### Notebook-to-Blog Workflow
1. Create/edit notebooks in `notebooks/` directory
2. Use conversion script: `python3 tools/convert_notebook.py notebooks/YourNotebook.ipynb`
3. Script automatically:
   - Generates Jekyll frontmatter with categories/tags
   - Converts notebook to markdown
   - Copies images to `assets/img/posts/[slug]/`
   - Updates image paths in markdown
4. Review generated post in `_posts/` directory
5. Test with `./tools/run.sh`

## Dependencies

- Ruby ~3.2 with Jekyll ~4.3.2
- Jekyll Chirpy theme and plugins
- HTML Proofer for testing
- Jupyter + nbconvert for notebook development
- Python 3 for conversion scripts