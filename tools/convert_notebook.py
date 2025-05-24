#!/usr/bin/env python3
"""
Convert Jupyter notebook to Jekyll blog post with proper asset handling.

Usage:
    python convert_notebook.py notebook.ipynb [--title "Custom Title"] [--date YYYY-MM-DD]
"""

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

def extract_frontmatter_from_notebook(notebook_path):
    """Extract metadata from notebook for Jekyll frontmatter."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Default values
    title = notebook_path.stem.replace('_', ' ').replace('-', ' ').title()
    categories = ["Machine Learning", "Tutorial"]
    tags = ["Python", "Data Science"]
    
    # Try to extract from first markdown cell if it contains frontmatter-like content
    if notebook.get('cells') and notebook['cells'][0].get('cell_type') == 'markdown':
        first_cell = ''.join(notebook['cells'][0].get('source', []))
        if 'categories:' in first_cell.lower() or 'tags:' in first_cell.lower():
            # Parse basic metadata from markdown
            if 'anomaly' in first_cell.lower():
                tags.extend(["Anomaly Detection", "PyTorch"])
            if 'transformer' in first_cell.lower():
                tags.extend(["Transformers", "Deep Learning"])
            if 'rule' in first_cell.lower():
                tags.extend(["Rule Mining", "Clinical Data"])
    
    return {
        'title': title,
        'categories': categories,
        'tags': tags
    }

def convert_notebook_to_markdown(notebook_path, output_dir, title=None, date=None):
    """Convert notebook to markdown and handle assets."""
    notebook_path = Path(notebook_path)
    output_dir = Path(output_dir)
    
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    # Generate output filename
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Extract metadata
    metadata = extract_frontmatter_from_notebook(notebook_path)
    if title:
        metadata['title'] = title
    
    # Create slug from title
    slug = re.sub(r'[^\w\s-]', '', metadata['title'].lower())
    slug = re.sub(r'[-\s]+', '-', slug).strip('-')
    
    output_filename = f"{date}-{slug}.md"
    output_path = output_dir / output_filename
    
    # Convert using nbconvert
    import subprocess
    
    # Convert to markdown
    cmd = [
        "jupyter", "nbconvert",
        "--to", "markdown",
        "--output-dir", str(output_dir),
        "--output", output_filename.replace('.md', ''),
        str(notebook_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting notebook: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return
    
    # Read the generated markdown
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create Jekyll frontmatter
    frontmatter = f"""---
title: {metadata['title']}
author: Deniz Akdemir
date: {date} 12:00:00 +0000
categories: {metadata['categories']}
tags: {metadata['tags']}
render_with_liquid: false
---

"""
    
    # Write back with frontmatter
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter + content)
    
    print(f"✅ Created blog post: {output_path}")
    
    # Handle image assets
    assets_dir = notebook_path.parent / f"{notebook_path.stem}_files"
    if assets_dir.exists():
        # Copy images to assets/img/posts/
        post_assets_dir = Path("assets/img/posts") / slug
        post_assets_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in assets_dir.glob("*"):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg', '.webp']:
                shutil.copy2(img_file, post_assets_dir)
                print(f"📷 Copied image: {img_file.name}")
        
        # Update image paths in markdown
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace image paths
        old_pattern = f"{notebook_path.stem}_files/"
        new_pattern = f"/assets/img/posts/{slug}/"
        content = content.replace(old_pattern, new_pattern)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"🔗 Updated image paths to: {new_pattern}")

def main():
    parser = argparse.ArgumentParser(description="Convert Jupyter notebook to Jekyll blog post")
    parser.add_argument("notebook", help="Path to the Jupyter notebook")
    parser.add_argument("--title", help="Custom title for the blog post")
    parser.add_argument("--date", help="Publication date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Convert notebook
    posts_dir = Path("_posts")
    posts_dir.mkdir(exist_ok=True)
    
    convert_notebook_to_markdown(args.notebook, posts_dir, args.title, args.date)

if __name__ == "__main__":
    main()