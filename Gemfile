# frozen_string_literal: true

source "https://rubygems.org"

# Use compatible versions for Ruby 3.2
gem "jekyll", "~> 4.3.2"
gem "jekyll-theme-chirpy", "~> 7.2.4"
gem "jekyll-include-cache"
gem "webrick"

group :jekyll_plugins do
  gem "jekyll-remote-theme"
  gem "jekyll-paginate"
  gem "jekyll-sitemap"
  gem "jekyll-gist"
  gem "jekyll-feed", "~> 0.12"
  gem "jemoji"
  gem "jekyll-seo-tag"
  gem "jekyll-archives"
  gem "jekyll-redirect-from"
end

gem "html-proofer", "~> 3.19.0", group: :test

platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds since newer versions of the gem
# do not have a Java counterpart.
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]
