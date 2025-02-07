---
layout: post
title: "Launching Your First GitHub Pages Blog: A Comprehensive Guide"
date: 2025-02-07
categories: [GitHub Pages, Jekyll, Blog Setup]
excerpt: "A step-by-step guide to creating, customizing, and launching your very own blog using GitHub Pages and Jekyll."
---

## Introduction

Welcome to your new digital space! In this guide, you will learn how to create a blog on GitHub Pages using Jekyll—a lightweight, static site generator that is both flexible and powerful. This tutorial incorporates essential steps, styling considerations, and configuration examples to ensure your new blog looks professional and runs smoothly.

## Step 1: Create a GitHub Repository

1. **Log in to GitHub:** Access your [GitHub account](https://github.com/).
2. **New Repository:** Click the **"+"** icon in the top right corner and select **"New repository"**.
3. **Repository Name:** Name it as `yourusername.github.io` (replace `yourusername` with your actual GitHub username).
4. **Public Access:** Ensure the repository is **public**.
5. **Initialize Repository:** Check the box to add a README.
6. **Create:** Click **"Create repository"**.

## Step 2: Enable GitHub Pages

1. **Navigate to Settings:** Go to your repository’s main page and click the **Settings** tab.
2. **Scroll Down to Pages:** In the left-hand menu, find **Pages** (or scroll down if using older GitHub UI).
3. **Set the Source:** Under **"Source"**, choose the `main` branch, then click **"Save"**.
4. **Verify Your URL:** GitHub will display your new site URL, typically:
   ```
   https://yourusername.github.io/
   ```
   Wait a few minutes for GitHub to generate the site. Then, visit the URL to confirm it is live.

## Step 3: Configure Jekyll via `_config.yml`

By default, GitHub Pages uses Jekyll to build your site. You can configure Jekyll by creating a `_config.yml` file in your repository’s root directory. Below is a comprehensive example:

```yaml
# Site settings
title: "Professional & Personal Blog"
description: "Insights on technology, coding, and life"
url: "https://yourusername.github.io"
baseurl: ""

# Author details
author:
  name: "Your Name"
  email: "your.email@example.com"

# Theme and plugins
theme: minima
plugins:
  - jekyll-feed
  - jekyll-seo-tag

# Markdown processing
markdown: kramdown
highlighter: rouge

# Permalink structure
permalink: /:year/:month/:day/:title/

# Sass folder configuration (for custom SCSS overrides)
sass:
  sass_dir: _sass

# Optional: pagination, timezone, etc.
paginate: 5
timezone: "UTC"
```

### Choosing a Different Theme
Instead of `minima`, you can try an official GitHub Pages theme like:
- `jekyll-theme-cayman`
- `jekyll-theme-slate`
- `jekyll-theme-midnight`
- `jekyll-theme-hacker`

Simply replace `theme: minima` with, for instance, `theme: jekyll-theme-cayman`.

## Step 4: Create a Custom SCSS File (Optional for Minima)

If you want to stay with **Minima** but prefer a different color scheme, font sizes, or layout tweaks, you can override the default styles with your own SCSS:

1. **Create Folders**:  
   ```
   _sass
   └── minima
       └── custom-styles.scss
   ```
2. **Add Overrides** in `custom-styles.scss`:
   ```scss
   /* Example overrides for Minima theme */

   /* Adjust the site title font size */
   .site-title {
     font-size: 1.75rem;
     font-weight: 700;
   }

   /* Hide the site description if you prefer a cleaner header */
   .site-description {
     display: none;
   }

   /* Tweak content width and colors */
   .page-content {
     max-width: 700px;
     margin: 0 auto;
     color: #333;
   }

   /* Modify link appearance */
   a {
     color: #007acc;
   }
   a:hover {
     color: #005ea3;
     text-decoration: underline;
   }
   ```

Minima automatically imports this file if `_config.yml` is configured correctly (`sass_dir: _sass` and `theme: minima`).

## Step 5: Write Your First Blog Post

To add your first post:

1. **Create a `_posts` Directory:** If it does not exist already.
2. **New Markdown File**: For example, `_posts/2025-02-07-first-post.md`.
3. **Front Matter and Content**: Below is a sample structure:

   ```markdown
   ---
   layout: post
   title: "Launching Your First GitHub Pages Blog: A Comprehensive Guide"
   date: 2025-02-07
   categories: [GitHub Pages, Jekyll, Blog Setup]
   excerpt: "A step-by-step guide to creating and launching your very own blog using GitHub Pages and Jekyll."
   ---

   ## Introduction
   
   Welcome to your new digital space! This guide walks you through creating a GitHub Pages blog using Jekyll—a lightweight, static site generator that is both flexible and powerful. Whether you are a seasoned developer or a beginner, this step-by-step tutorial will help you set up a blog that is visually appealing and easy to maintain.

   ## Step 1: Create a GitHub Repository
   ...

   ## Step 2: Enable GitHub Pages
   ...

   (Continue with your detailed instructions here)
   ```

4. **Commit and Verify**: Once you commit this file, wait a few moments and refresh your blog URL. You should see your first post listed.

## Step 6: Customize the Homepage (Optional)

If you notice repeated headings or want a unique homepage:

1. **Create an `index.md`** file at the root of your repository.
2. **Use Front Matter** to specify a layout:
   ```markdown
   ---
   layout: home
   title: ""
   ---

   # Welcome to My Blog
   This is a custom landing page for my professional and personal posts.
   ```
   - Setting `title: ""` or `title: false` can reduce repeated site titles.
   - You can also create a completely custom layout by editing or adding files under `_layouts/`.

## Step 7: Explore Further Customizations

- **Set a Custom Domain**: In **Settings → Pages**, you can add a custom domain (e.g., `www.yourdomain.com`).  
- **Add More Plugins**: Explore [Jekyll plugins](https://jekyllrb.com/docs/plugins/) to integrate search, analytics, or additional features.  
- **Experiment with Theme Configuration**: Many official themes allow further customization through `_config.yml` variables.

## Conclusion

By combining the steps above with optional SCSS overrides and theme changes, you can quickly transform a basic GitHub Pages site into a polished, professional blog. Feel free to iterate, experiment with various layouts, and tailor your blog’s design to suit your personal style or branding needs. Once you are satisfied with the appearance, you can focus on creating regular content and sharing your ideas with the world.

Happy blogging!
