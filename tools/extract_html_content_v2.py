#!/usr/bin/env python3
"""
Extract main content from HTML file preserving R code, outputs, and figures.
Version 2 with improved parsing and image handling.
"""
import re
import os
import sys
import base64
import json
from pathlib import Path
from bs4 import BeautifulSoup
import argparse

class HTMLContentExtractor:
    def __init__(self, input_file, output_file=None, images_dir=None):
        self.input_path = Path(input_file)
        self.output_path = Path(output_file) if output_file else self.input_path.with_suffix('.md')
        self.images_dir = Path(images_dir) if images_dir else self.output_path.parent / f"{self.output_path.stem}_images"
        self.image_counter = 0
        self.extracted_images = {}
        
    def extract_base64_images(self, soup):
        """Extract all base64 encoded images and save them as files."""
        images = soup.find_all('img', src=re.compile(r'^data:image'))
        
        for img in images:
            src = img.get('src', '')
            match = re.match(r'data:image/([^;]+);base64,(.+)', src)
            if match:
                img_type, base64_data = match.groups()
                self.image_counter += 1
                
                # Create filename
                filename = f"figure_{self.image_counter}.{img_type}"
                filepath = self.images_dir / filename
                
                # Decode and save image
                try:
                    img_data = base64.b64decode(base64_data)
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                    
                    # Update the image src to point to the file
                    relative_path = f"{self.output_path.stem}_images/{filename}"
                    img['src'] = relative_path
                    self.extracted_images[src[:100]] = relative_path
                    print(f"Extracted image: {filename}")
                except Exception as e:
                    print(f"Error extracting image {self.image_counter}: {e}")
    
    def convert_code_block(self, element):
        """Convert code blocks to markdown format."""
        code_text = element.get_text().strip()
        
        # Check if it's R code
        classes = element.get('class', [])
        if isinstance(classes, str):
            classes = classes.split()
        
        if any('r' in cls.lower() for cls in classes):
            return f"\n```r\n{code_text}\n```\n"
        elif any('sourceCode' in cls for cls in classes):
            return f"\n```r\n{code_text}\n```\n"
        else:
            return f"\n```\n{code_text}\n```\n"
    
    def convert_table(self, table):
        """Convert HTML table to Markdown table."""
        rows = table.find_all('tr')
        if not rows:
            return ""
        
        markdown_lines = []
        
        # Process header row
        header_cells = rows[0].find_all(['th', 'td'])
        if header_cells:
            headers = [cell.get_text().strip() for cell in header_cells]
            markdown_lines.append('| ' + ' | '.join(headers) + ' |')
            markdown_lines.append('|' + '---|' * len(headers))
        
        # Process data rows
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if cells:
                cell_texts = [cell.get_text().strip() for cell in cells]
                markdown_lines.append('| ' + ' | '.join(cell_texts) + ' |')
        
        return '\n' + '\n'.join(markdown_lines) + '\n'
    
    def extract_content(self, soup):
        """Extract the main content from the HTML."""
        content = []
        
        # Look for main content container
        main_container = (
            soup.find('div', {'class': 'container-fluid main-container'}) or
            soup.find('div', {'id': 'notebook'}) or
            soup.find('div', {'class': 'notebook'}) or
            soup.find('main') or
            soup.find('article') or
            soup.find('body')
        )
        
        if not main_container:
            print("Warning: Could not find main content container")
            main_container = soup
        
        # Process all relevant elements
        for element in main_container.descendants:
            if element.name is None:  # Text node
                continue
                
            # Skip if element is within a script or style tag
            if element.find_parent(['script', 'style']):
                continue
            
            # Headers
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1])
                header_text = element.get_text().strip()
                if header_text:
                    content.append('\n' + '#' * level + ' ' + header_text + '\n')
            
            # Paragraphs
            elif element.name == 'p':
                text = element.get_text().strip()
                if text:
                    content.append(text + '\n')
            
            # Code blocks
            elif element.name == 'pre':
                # Skip if already processed as part of a parent div
                parent_div = element.find_parent('div', class_='sourceCode')
                if parent_div:
                    continue
                
                code = element.find('code')
                if code:
                    content.append(self.convert_code_block(code))
                else:
                    content.append(self.convert_code_block(element))
            
            # Source code divs
            elif element.name == 'div' and 'sourceCode' in element.get('class', []):
                pre = element.find('pre')
                if pre:
                    code = pre.find('code')
                    if code:
                        content.append(self.convert_code_block(code))
            
            # Output divs
            elif element.name == 'div' and element.get('class'):
                classes = element.get('class', [])
                if any('output' in cls for cls in classes):
                    # Skip if this is a parent container for other outputs
                    if element.find_all('div', class_=re.compile('output')):
                        continue
                    
                    output_text = element.get_text().strip()
                    if output_text:
                        content.append(f"\n```\n{output_text}\n```\n")
            
            # Images
            elif element.name == 'img':
                src = element.get('src', '')
                alt = element.get('alt', 'Image')
                if src:
                    content.append(f"\n![{alt}]({src})\n")
            
            # Tables
            elif element.name == 'table':
                # Skip if already processed as child of another element
                if any(element in child.descendants for child in content):
                    continue
                content.append(self.convert_table(element))
        
        return '\n'.join(content)
    
    def run(self):
        """Main extraction process."""
        # Create images directory
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Read HTML file
        print(f"Reading {self.input_path}...")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse HTML
        print("Parsing HTML content...")
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract base64 images
        print("Extracting embedded images...")
        self.extract_base64_images(soup)
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else self.input_path.stem
        
        # Extract meta information if available
        meta_info = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name', '')
            content = meta.get('content', '')
            if name and content:
                meta_info[name] = content
        
        # Convert to markdown
        print("Converting to Markdown...")
        markdown_content = self.extract_content(soup)
        
        # Create front matter
        front_matter = f"""---
title: "{title_text}"
layout: post
date: {meta_info.get('date', 'YYYY-MM-DD')}
categories: [R, Finance, Statistics]
tags: [portfolio-optimization, covariance-estimation, mixed-models]
---

"""
        
        # Write output
        print(f"Writing to {self.output_path}...")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(front_matter + markdown_content)
        
        print(f"\nExtraction complete!")
        print(f"Markdown file: {self.output_path}")
        print(f"Images directory: {self.images_dir}")
        print(f"Total images extracted: {len(self.extracted_images)}")
        
        return self.output_path

def main():
    parser = argparse.ArgumentParser(description='Extract content from HTML file')
    parser.add_argument('input_file', help='Input HTML file')
    parser.add_argument('--output', '-o', help='Output markdown file', default=None)
    parser.add_argument('--images-dir', help='Directory for extracted images', default=None)
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)
    
    extractor = HTMLContentExtractor(args.input_file, args.output, args.images_dir)
    extractor.run()

if __name__ == '__main__':
    main()