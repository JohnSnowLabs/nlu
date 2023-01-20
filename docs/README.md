# NLU Documentation

We welcome you to contribute to NLU documentation hosted inside `en/` directory. All the files are in Markdown format.

## Install
[You need to install Jekyll](https://jekyllrb.com/docs/installation/ubuntu/)

## Development

For development purposes, you need to have `bundle` and `Gem` installed on your system. Please run these commands:

```bash
bundle update
bundle install
bundle exec jekyll serve

# Server address: http://127.0.0.1:4000
```

## Troubleshooting

If you get this error after `bundle update`:

```
In Gemfile:
  github-pages was resolved to 227, which depends on
    jekyll-mentions was resolved to 1.6.0, which depends on
      html-pipeline was resolved to 2.14.2, which depends on
        nokogiri
```

Just do `rm Gemfile.lock` and rerun again `bundle update`.
