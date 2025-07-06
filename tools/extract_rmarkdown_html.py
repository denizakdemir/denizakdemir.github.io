#!/usr/bin/env python3
"""
Extract content from R Markdown HTML output.
Specialized for handling R code chunks, outputs, and embedded images.
"""
import re
import os
import sys
import base64
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString
import argparse

def extract_and_save_base64_image(img_src, img_num, images_dir):
    """Extract a base64 image and save it to disk."""
    match = re.match(r'data:image/([^;]+);base64,(.+)', img_src)
    if not match:
        return None
    
    img_type, base64_data = match.groups()
    filename = f"figure_{img_num}.{img_type}"
    filepath = images_dir / filename
    
    try:
        img_data = base64.b64decode(base64_data)
        with open(filepath, 'wb') as f:
            f.write(img_data)
        return filename
    except Exception as e:
        print(f"Error saving image {img_num}: {e}")
        return None

def process_element(element, images_dir, img_counter, content_parts, processed_elements):
    """Process a single HTML element and convert to markdown."""
    # Skip if already processed
    if element in processed_elements:
        return img_counter
    
    processed_elements.add(element)
    
    # Headers
    if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        level = int(element.name[1])
        text = element.get_text().strip()
        if text:
            content_parts.append('\n' + '#' * level + ' ' + text + '\n')
    
    # Paragraphs
    elif element.name == 'p':
        text = element.get_text().strip()
        if text:
            content_parts.append(text + '\n')
    
    # Code chunks (R input)
    elif element.name == 'div' and 'sourceCode' in element.get('class', []):
        pre = element.find('pre')
        if pre:
            code = pre.find('code')
            if code:
                code_text = code.get_text().strip()
                content_parts.append('\n```r\n' + code_text + '\n```\n')
                # Mark all children as processed
                for child in element.descendants:
                    processed_elements.add(child)
    
    # Pre elements (could be code or output)
    elif element.name == 'pre':
        # Check if it's inside a sourceCode div (already processed)
        if element.find_parent('div', class_='sourceCode'):
            return img_counter
        
        # Check if it has a code child
        code = element.find('code')
        if code:
            code_text = code.get_text().strip()
            # Check if it looks like R code (has R-specific patterns)
            if any(pattern in code_text for pattern in ['<-', 'function(', 'library(', '~', 'data.frame']):
                content_parts.append('\n```r\n' + code_text + '\n```\n')
            else:
                content_parts.append('\n```\n' + code_text + '\n```\n')
        else:
            # Plain output
            text = element.get_text().strip()
            if text:
                content_parts.append('\n```\n' + text + '\n```\n')
    
    # Images
    elif element.name == 'img':
        src = element.get('src', '')
        alt = element.get('alt', 'Image')
        
        if src.startswith('data:image'):
            img_counter += 1
            filename = extract_and_save_base64_image(src, img_counter, images_dir)
            if filename:
                relative_path = f"{images_dir.name}/{filename}"
                content_parts.append(f'\n![{alt}]({relative_path})\n')
                print(f"Extracted image: {filename}")
        elif src:
            content_parts.append(f'\n![{alt}]({src})\n')
    
    # Tables
    elif element.name == 'table':
        rows = element.find_all('tr')
        if rows:
            table_lines = []
            
            # Header row
            header_cells = rows[0].find_all(['th', 'td'])
            if header_cells:
                headers = [cell.get_text().strip() for cell in header_cells]
                table_lines.append('| ' + ' | '.join(headers) + ' |')
                table_lines.append('|' + '---|' * len(headers))
            
            # Data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if cells:
                    cell_texts = [cell.get_text().strip() for cell in cells]
                    table_lines.append('| ' + ' | '.join(cell_texts) + ' |')
            
            if table_lines:
                content_parts.append('\n' + '\n'.join(table_lines) + '\n')
        
        # Mark all children as processed
        for child in element.descendants:
            processed_elements.add(child)
    
    # Lists
    elif element.name in ['ul', 'ol']:
        list_items = element.find_all('li', recursive=False)
        for li in list_items:
            text = li.get_text().strip()
            if text:
                if element.name == 'ul':
                    content_parts.append('- ' + text + '\n')
                else:
                    content_parts.append('1. ' + text + '\n')
        content_parts.append('\n')
        
        # Mark all children as processed
        for child in element.descendants:
            processed_elements.add(child)
    
    return img_counter

def extract_rmarkdown_content(html_file, output_file=None, images_dir=None):
    """Main extraction function."""
    input_path = Path(html_file)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        return None
    
    # Set up paths
    output_path = Path(output_file) if output_file else input_path.with_suffix('.md')
    images_path = Path(images_dir) if images_dir else output_path.parent / f"{output_path.stem}_images"
    
    # Create images directory
    images_path.mkdir(parents=True, exist_ok=True)
    
    # Read and parse HTML
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("Parsing HTML...")
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract title
    title = soup.find('title')
    title_text = title.get_text().strip() if title else input_path.stem
    
    # Find main content container
    main_content = (
        soup.find('div', {'class': 'container-fluid main-container'}) or
        soup.find('div', {'id': 'notebook-container'}) or
        soup.find('main') or
        soup.find('article') or
        soup.body
    )
    
    if not main_content:
        print("Warning: Could not find main content container, using full body")
        main_content = soup
    
    # Process content
    print("Extracting content...")
    content_parts = []
    img_counter = 0
    processed_elements = set()
    
    # Process all direct children and their descendants
    for element in main_content.children:
        if isinstance(element, NavigableString):
            continue
        
        # Process this element
        img_counter = process_element(element, images_path, img_counter, content_parts, processed_elements)
        
        # Process descendants if not already processed
        if hasattr(element, 'descendants'):
            for desc in element.descendants:
                if isinstance(desc, NavigableString):
                    continue
                if desc not in processed_elements:
                    img_counter = process_element(desc, images_path, img_counter, content_parts, processed_elements)
    
    # Create markdown content
    markdown_content = ''.join(content_parts)
    
    # Create front matter
    front_matter = f"""---
title: "{title_text}"
layout: post
date: {Path(html_file).stat().st_mtime}
categories: [R, Finance, Statistics]
tags: [portfolio-optimization, covariance-estimation, mixed-models]
render_with_liquid: false
---

"""
    
    # Write output
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(front_matter + markdown_content)
    
    print(f"\nExtraction complete!")
    print(f"Output file: {output_path}")
    print(f"Images saved to: {images_path}")
    print(f"Total images extracted: {img_counter}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Extract content from R Markdown HTML')
    parser.add_argument('input_file', help='Input HTML file')
    parser.add_argument('--output', '-o', help='Output markdown file')
    parser.add_argument('--images-dir', help='Directory for extracted images')
    
    args = parser.parse_args()
    
    extract_rmarkdown_content(args.input_file, args.output, args.images_dir)

if __name__ == '__main__':
    main()