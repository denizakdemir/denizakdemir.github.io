#!/usr/bin/env bash
#
# Blog setup and maintenance script
# Usage: ./tools/setup_blog.sh [clean|convert|setup]

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

show_help() {
    echo "Blog Setup and Maintenance Script"
    echo
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  clean     - Clean up duplicate files and build artifacts"
    echo "  convert   - Convert all notebooks to blog posts (interactive)"
    echo "  setup     - Install dependencies and setup development environment"
    echo "  status    - Show repository status and recommendations"
    echo "  help      - Show this help message"
    echo
}

clean_repo() {
    echo "🧹 Cleaning repository..."
    
    # Remove build artifacts
    [ -d "_site" ] && rm -rf "_site" && echo "  ✅ Removed _site/"
    
    # Create assets directory structure
    mkdir -p "assets/img/posts"
    echo "  ✅ Created assets/img/posts/"
    
    # Check for duplicate image directories
    if [ -d "AnomalyDetection_files" ] || [ -d "HierarchyRules_files" ]; then
        echo "  ⚠️  Found duplicate image directories (already cleaned)"
    fi
    
    echo "✅ Repository cleaned"
}

setup_environment() {
    echo "⚙️  Setting up development environment..."
    
    # Check if bundle is available
    if ! command -v bundle &> /dev/null; then
        echo "❌ Bundle not found. Please install Ruby and bundler first."
        echo "   macOS: brew install ruby"
        echo "   Then: gem install bundler"
        exit 1
    fi
    
    # Install Ruby dependencies
    echo "  📦 Installing Ruby dependencies..."
    bundle install
    
    # Check if jupyter is available for notebook conversion
    if command -v jupyter &> /dev/null; then
        echo "  ✅ Jupyter found - notebook conversion available"
    else
        echo "  ⚠️  Jupyter not found - install with: pip install jupyter nbconvert"
    fi
    
    echo "✅ Development environment ready"
}

show_status() {
    echo "📊 Repository Status"
    echo "==================="
    
    # Count blog posts
    post_count=$(find _posts -name "*.md" 2>/dev/null | wc -l)
    echo "📝 Blog posts: $post_count"
    
    # Count notebooks
    notebook_count=$(find notebooks -name "*.ipynb" 2>/dev/null | wc -l)
    echo "📓 Notebooks: $notebook_count"
    
    # Check for missing blog posts
    echo
    echo "🔍 Notebook Analysis:"
    for notebook in notebooks/*.ipynb; do
        [ -f "$notebook" ] || continue
        basename_nb=$(basename "$notebook" .ipynb)
        
        # Look for corresponding blog post (more flexible matching)
        basename_lower=$(echo "$basename_nb" | tr '[:upper:]' '[:lower:]')
        
        # Try multiple matching patterns
        has_post=false
        case "$basename_nb" in
            "AnomalyDetection")
                if ls _posts/*"anomaly-detection"* &>/dev/null; then has_post=true; fi
                ;;
            "TransformerExample")
                if ls _posts/*"tabular-transformer"* &>/dev/null; then has_post=true; fi
                ;;
            "HierarchyRules")
                if ls _posts/*"rule-mining"* &>/dev/null || ls _posts/*"hierarchy"* &>/dev/null; then has_post=true; fi
                ;;
            "AnomalyDetectionGAN")
                if ls _posts/*"gan"* &>/dev/null; then has_post=true; fi
                ;;
            *)
                if ls _posts/*"$basename_lower"* &>/dev/null; then has_post=true; fi
                ;;
        esac
        
        if $has_post; then
            echo "  ✅ $basename_nb - has blog post"
        else
            echo "  ❌ $basename_nb - missing blog post"
        fi
    done
    
    # Check git status
    echo
    echo "📦 Git Status:"
    if [ -n "$(git status --porcelain)" ]; then
        echo "  ⚠️  Uncommitted changes found"
        git status --short
    else
        echo "  ✅ Working directory clean"
    fi
}

convert_notebooks() {
    echo "🔄 Converting notebooks to blog posts..."
    
    # Check if conversion script exists and jupyter is available
    if [ ! -f "tools/convert_notebook.py" ]; then
        echo "❌ Conversion script not found"
        exit 1
    fi
    
    if ! command -v jupyter &> /dev/null; then
        echo "❌ Jupyter not found. Install with: pip install jupyter nbconvert"
        exit 1
    fi
    
    # Find notebooks without corresponding blog posts
    missing_posts=()
    for notebook in notebooks/*.ipynb; do
        [ -f "$notebook" ] || continue
        basename_nb=$(basename "$notebook" .ipynb)
        
        # Check if blog post exists (using same logic as status)
        has_post=false
        case "$basename_nb" in
            "AnomalyDetection")
                if ls _posts/*"anomaly-detection"* &>/dev/null; then has_post=true; fi
                ;;
            "TransformerExample")
                if ls _posts/*"tabular-transformer"* &>/dev/null; then has_post=true; fi
                ;;
            "HierarchyRules")
                if ls _posts/*"rule-mining"* &>/dev/null || ls _posts/*"hierarchy"* &>/dev/null; then has_post=true; fi
                ;;
            "AnomalyDetectionGAN")
                if ls _posts/*"gan"* &>/dev/null; then has_post=true; fi
                ;;
            *)
                basename_lower=$(echo "$basename_nb" | tr '[:upper:]' '[:lower:]')
                if ls _posts/*"$basename_lower"* &>/dev/null; then has_post=true; fi
                ;;
        esac
        
        if ! $has_post; then
            missing_posts+=("$notebook")
        fi
    done
    
    if [ ${#missing_posts[@]} -eq 0 ]; then
        echo "✅ All notebooks have corresponding blog posts"
        return
    fi
    
    echo "Found ${#missing_posts[@]} notebooks without blog posts:"
    for notebook in "${missing_posts[@]}"; do
        echo "  - $(basename "$notebook")"
    done
    
    echo
    read -p "Convert all missing notebooks? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for notebook in "${missing_posts[@]}"; do
            echo "Converting $(basename "$notebook")..."
            python3 tools/convert_notebook.py "$notebook"
        done
        echo "✅ Conversion complete"
    fi
}

# Main script logic
case "${1:-help}" in
    clean)
        clean_repo
        ;;
    setup)
        setup_environment
        ;;
    convert)
        convert_notebooks
        ;;
    status)
        show_status
        ;;
    help|*)
        show_help
        ;;
esac