#!/usr/bin/env python3
"""
Convert R Markdown HTML output to Jekyll blog post.
"""

import re
from pathlib import Path
from bs4 import BeautifulSoup
import html2text

def convert_rmd_html_to_jekyll(html_path, output_path, date="2025-07-05"):
    """Convert R Markdown HTML to Jekyll post format."""
    
    # Read HTML file
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract title
    title = soup.find('title').text if soup.find('title') else "Portfolio Optimization"
    
    # Extract main content
    # R Markdown usually puts content in a container div
    content_div = soup.find('div', class_='container-fluid main-container') or \
                  soup.find('div', class_='container') or \
                  soup.find('body')
    
    if content_div:
        # Initialize html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.body_width = 0  # Don't wrap lines
        
        # Convert to markdown
        markdown_content = h.handle(str(content_div))
        
        # Clean up the markdown
        # Remove any style tags
        markdown_content = re.sub(r'<style[^>]*>.*?</style>', '', markdown_content, flags=re.DOTALL)
        # Remove script tags
        markdown_content = re.sub(r'<script[^>]*>.*?</script>', '', markdown_content, flags=re.DOTALL)
        
        # Create Jekyll frontmatter
        frontmatter = f"""---
title: {title}
author: Deniz Akdemir
date: {date} 12:00:00 +0000
categories: [Finance, Tutorial]
tags: [Portfolio Optimization, Mixed Models, R, Finance, Quantitative Finance]
render_with_liquid: false
---

"""
        
        # Write the Jekyll post
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(frontmatter + markdown_content)
        
        print(f"✅ Created Jekyll post: {output_path}")
    else:
        print("❌ Could not find main content in HTML file")

if __name__ == "__main__":
    html_path = Path("notebooks/PortfolioOptimization.html")
    output_path = Path("_posts/2025-07-05-portfolio-optimization.md")
    convert_rmd_html_to_jekyll(html_path, output_path)