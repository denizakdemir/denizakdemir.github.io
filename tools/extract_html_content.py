#!/usr/bin/env python3
"""
Extract main content from HTML file preserving R code, outputs, and figures.
"""
import re
import os
import sys
import base64
from pathlib import Path
from bs4 import BeautifulSoup
import argparse

def extract_base64_images(html_content, output_dir):
    """Extract base64 encoded images and save them as files."""
    img_pattern = r'<img[^>]+src="data:image/([^;]+);base64,([^"]+)"[^>]*>'
    matches = re.findall(img_pattern, html_content)
    
    image_mapping = {}
    for i, (img_type, base64_data) in enumerate(matches):
        # Create filename
        filename = f"figure_{i + 1}.{img_type}"
        filepath = os.path.join(output_dir, filename)
        
        # Decode and save image
        img_data = base64.b64decode(base64_data)
        with open(filepath, 'wb') as f:
            f.write(img_data)
        
        # Store mapping of base64 data to filename
        image_mapping[base64_data[:100]] = filename  # Use first 100 chars as key
        print(f"Extracted image: {filename}")
    
    return image_mapping

def convert_html_to_markdown(soup, image_dir_relative):
    """Convert HTML content to Markdown while preserving structure."""
    content = []
    
    # Find the main content area
    main_content = soup.find('div', {'class': 'container-fluid main-container'})
    if not main_content:
        main_content = soup.find('body')
    
    for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'div', 'img', 'table']):
        if element.name.startswith('h'):
            level = int(element.name[1])
            content.append('\n' + '#' * level + ' ' + element.get_text().strip() + '\n')
        
        elif element.name == 'p':
            text = element.get_text().strip()
            if text:
                content.append(text + '\n')
        
        elif element.name == 'pre':
            # Handle code blocks
            code_element = element.find('code')
            if code_element:
                # Check if it's R code
                classes = code_element.get('class', [])
                if any('r' in cls for cls in classes):
                    content.append('\n```r\n' + code_element.get_text().strip() + '\n```\n')
                else:
                    content.append('\n```\n' + code_element.get_text().strip() + '\n```\n')
            else:
                content.append('\n```\n' + element.get_text().strip() + '\n```\n')
        
        elif element.name == 'div':
            # Check for output divs
            if 'sourceCode' in element.get('class', []):
                pre = element.find('pre')
                if pre:
                    code = pre.find('code')
                    if code:
                        content.append('\n```r\n' + code.get_text().strip() + '\n```\n')
            elif element.get('class') and any('output' in cls for cls in element.get('class', [])):
                # This is output
                text = element.get_text().strip()
                if text:
                    content.append('\n```\n' + text + '\n```\n')
        
        elif element.name == 'img':
            # Handle images
            src = element.get('src', '')
            alt = element.get('alt', 'Image')
            
            if src.startswith('data:image'):
                # Extract image type and find corresponding file
                match = re.match(r'data:image/([^;]+);base64,(.+)', src)
                if match:
                    # Find the saved image file based on the match
                    img_type = match.group(1)
                    # Count existing figures to get the right number
                    existing_figures = [f for f in content if 'figure_' in f]
                    figure_num = len(existing_figures) + 1
                    img_filename = f'figure_{figure_num}.{img_type}'
                    content.append(f'\n![{alt}]({image_dir_relative}/{img_filename})\n')
            else:
                content.append(f'\n![{alt}]({src})\n')
        
        elif element.name == 'table':
            # Convert HTML table to Markdown table
            rows = element.find_all('tr')
            if rows:
                # Header row
                headers = rows[0].find_all(['th', 'td'])
                if headers:
                    header_text = '| ' + ' | '.join(h.get_text().strip() for h in headers) + ' |'
                    separator = '|' + '---|' * len(headers)
                    content.append('\n' + header_text + '\n' + separator + '\n')
                    
                    # Data rows
                    for row in rows[1:]:
                        cells = row.find_all(['td', 'th'])
                        if cells:
                            row_text = '| ' + ' | '.join(cell.get_text().strip() for cell in cells) + ' |'
                            content.append(row_text + '\n')
                content.append('\n')
    
    return '\n'.join(content)

def main():
    parser = argparse.ArgumentParser(description='Extract content from HTML file')
    parser.add_argument('input_file', help='Input HTML file')
    parser.add_argument('--output', '-o', help='Output markdown file', default=None)
    parser.add_argument('--images-dir', help='Directory for extracted images', default=None)
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
    
    # Set output paths
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.md')
    
    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        images_dir = output_path.parent / f"{output_path.stem}_images"
    
    # Create images directory
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Read HTML file
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract base64 images
    print("Extracting embedded images...")
    image_mapping = extract_base64_images(html_content, str(images_dir))
    
    # Parse HTML
    print("Parsing HTML content...")
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract title
    title = soup.find('title')
    if title:
        title_text = title.get_text().strip()
    else:
        title_text = input_path.stem
    
    # Convert to markdown
    print("Converting to Markdown...")
    relative_images_dir = f"{output_path.stem}_images"
    markdown_content = convert_html_to_markdown(soup, relative_images_dir)
    
    # Add front matter
    front_matter = f"""---
title: "{title_text}"
layout: post
---

"""
    
    # Write output
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(front_matter + markdown_content)
    
    print(f"\nExtraction complete!")
    print(f"Markdown file: {output_path}")
    print(f"Images directory: {images_dir}")
    print(f"Total images extracted: {len(image_mapping)}")

if __name__ == '__main__':
    main()