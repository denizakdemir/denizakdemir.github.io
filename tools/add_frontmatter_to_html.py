#!/usr/bin/env python3
"""
Add Jekyll front matter to HTML file
"""

def add_frontmatter_to_html(html_file_path):
    # Read the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Jekyll front matter
    frontmatter = """---
title: "How to Use Mixed Models to Improve Portfolio Performance"
author: Deniz Akdemir
date: 2025-07-05 12:00:00 +0000
categories: [Finance, Machine Learning]
tags: [portfolio-optimization, mixed-models, covariance-estimation, R, quantitative-finance]
layout: null
---
"""
    
    # Write back with front matter
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter + content)
    
    print(f"✅ Added Jekyll front matter to {html_file_path}")

if __name__ == "__main__":
    add_frontmatter_to_html("_posts/2025-07-05-portfolio-optimization.html")