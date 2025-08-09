# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a SAS programming tutorial project designed to create a comprehensive multi-part blog series. The project consists of a 13-part tutorial series covering SAS programming from beginner to advanced levels, with an optional HTML-based tutorial website structure. This project is part of a larger Jekyll-based blog (denizakdemir.github.io) focused on data science and machine learning.

## Project Structure

- `sas_blog_series_outline.md` - Complete 13-part blog series outline with detailed content structure
- `html_blog_instructions.md` - Instructions for creating a multi-page HTML tutorial website
- `html/` - HTML tutorial website files (if implementing standalone version)
- Blog posts are placed in parent repository's `../../_posts/` directory

## Development Commands

### Local Development and Testing
- `../../tools/run.sh` - Start Jekyll development server to preview blog posts
- `../../tools/test.sh` - Build site and run HTML validation tests
- `../../tools/setup_blog.sh status` - Check repository health and missing posts
- `bundle exec jekyll build` - Build site manually from parent directory

### Blog Post Creation
- For markdown posts: Create directly in `../../_posts/` with format `YYYY-MM-DD-title.md`
- For notebook conversion: `python3 ../../tools/convert_notebook.py notebooks/YourNotebook.ipynb`
- Ensure Jekyll front matter includes `render_with_liquid: false` for technical content

### HTML Tutorial Development
If building the HTML version:
- Create directory structure as specified in `html_blog_instructions.md`
- Include responsive design and mobile compatibility
- Implement syntax highlighting for SAS code blocks
- Add copy-to-clipboard functionality for code examples

## Architecture

### Blog Series Structure
1. **Part 1**: Getting Started with SAS (Beginner, 2000-2500 words)
2. **Part 2**: SAS Fundamentals (Beginner, 2500-3000 words)
3. **Part 3**: Mastering the DATA Step (Intermediate, 3500-4000 words)
4. **Part 4**: Variables, Formats, and Data Types (Intermediate, 2500-3000 words)
5. **Part 5**: Data Input and Output (Intermediate, 3000-3500 words)
6. **Part 6**: Data Manipulation and Processing (Intermediate, 3500-4000 words)
7. **Part 7**: Control Structures and Loops (Intermediate, 3000-3500 words)
8. **Part 8**: SAS Functions (Intermediate, 4000-4500 words)
9. **Part 9**: Essential PROC Steps (Intermediate, 4000-4500 words)
10. **Part 10**: SQL in SAS (Intermediate, 3500-4000 words)
11. **Part 11**: Debugging and Error Handling (Advanced, 2500-3000 words)
12. **Part 12**: Advanced Topics and Best Practices (Advanced, 3000-3500 words)
13. **Part 13**: SAS Certification Preparation (All levels, 2000-2500 words)

### Content Guidelines
- Each post includes: Introduction, Prerequisites, Main Content, Practical Exercises, Summary, Next Steps
- Target specific SAS-related keywords for SEO
- Include 5-10 code examples per tutorial
- Provide downloadable practice datasets and code files
- Use consistent SAS code formatting and commenting style

### SAS Code Standards
- Use proper indentation in DATA and PROC steps
- Include descriptive comments for complex logic
- Follow SAS naming conventions for variables and datasets
- Provide error handling examples where appropriate

## Technical Specifications

### For Blog Posts
- Categories: `[SAS Programming, Tutorial]`
- Tags: Based on topic (e.g., `[DATA Step, PROC SQL, Functions, Debugging]`)
- Code blocks should use `sas` language identifier for syntax highlighting
- Include output examples where relevant

### For HTML Tutorial
- Responsive grid layout with mobile-first design
- SAS syntax highlighting using Prism.js or similar
- Interactive features: code copy buttons, progress tracking, search
- SEO optimization with proper meta tags and structured data

## Development Workflow

1. **Content Creation**
   - Draft content following the outline structure
   - Test all SAS code examples in SAS OnDemand or similar environment
   - Create sample datasets for exercises
   - Generate screenshots of SAS Studio interface where helpful

2. **Blog Post Generation**
   - Convert content to markdown format
   - Add appropriate Jekyll front matter
   - Place in `../../_posts/` directory with naming: `YYYY-MM-DD-sas-tutorial-part-N.md`
   - Test rendering with `../../tools/run.sh`

3. **Quality Assurance**
   - Verify all code examples execute without errors
   - Check for consistent formatting and style
   - Ensure logical progression between tutorials
   - Test all download links and resources
   - Run `../../tools/test.sh` to validate HTML output

## Current Progress

- **Completed**: Parts 1-2 of the blog series
  - `2025-08-09-getting-started-with-sas.md`
  - `2025-08-10-sas-fundamentals.md`
- **Remaining**: Parts 3-13 to be created following the outline
- **HTML Tutorial**: Optional standalone version partially structured