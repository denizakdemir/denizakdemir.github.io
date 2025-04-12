# frozen_string_literal: true

source "https://rubygems.org"

# Use compatible versions for Ruby 2.6
gem "jekyll", "~> 4.3.0"
gem "jekyll-theme-chirpy", "~> 5.6.0"  # Using an older version compatible with Ruby 2.6

gem "html-proofer", "~> 3.19.0", group: :test  # Older version compatible with Ruby 2.6

platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]
