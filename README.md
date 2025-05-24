# Deniz Akdemir's Blog

A personal and professional blog focused on data science, machine learning, and statistical analysis. The blog covers various advanced topics including:

## Featured Topics
- Anomaly Detection using Deep Learning (Autoencoders, GANs)
- Transformer Models for Tabular Data
- Rule Mining in Clinical Data
- Statistical Analysis and Data Science

## Technical Stack
- Built with Jekyll
- Includes Jupyter notebooks with practical implementations
- Contains datasets and example code for demonstrations

## Content Structure
- Blog posts with detailed technical explanations
- Interactive Jupyter notebooks
- Example datasets and models
- Professional information and contact details

## Local Development

### Quick Start
```bash
# Setup development environment
./tools/setup_blog.sh setup

# Start development server
./tools/run.sh

# Check repository status
./tools/setup_blog.sh status
```

### Notebook to Blog Workflow
1. Create/edit Jupyter notebooks in the `notebooks/` directory
2. Convert to blog posts: `python3 tools/convert_notebook.py notebooks/YourNotebook.ipynb`
3. Review generated markdown in `_posts/`
4. Test locally with `./tools/run.sh`

### Available Commands
- `./tools/run.sh` - Start Jekyll development server with live reload
- `./tools/test.sh` - Build site for production and run tests
- `./tools/setup_blog.sh status` - Show repository status and missing blog posts
- `./tools/setup_blog.sh clean` - Clean up duplicate files and build artifacts
- `./tools/setup_blog.sh convert` - Interactive notebook conversion

Visit [denizakdemir.github.io](https://denizakdemir.github.io) to explore the blog.
