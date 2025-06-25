# MakeSenseOfIt

MakeSenseOfIt is a command line tool for scraping posts from Reddit and Twitter,
performing sentiment analysis and generating rich reports.  The application is
split into small modules (scrapers, sentiment, processing, export and
visualisation) so it can easily be extended or used as a library.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All run time options are defined in a JSON configuration file.  By default the
application loads `config.json` from the project root.  A sample file is provided
as `config.json.example`.

Important keys are:

- `queries` – list of search terms
- `default_platforms` – platforms to scrape (`twitter`, `reddit`)
- `default_time_window` – number of days to look back
- `output_formats` – formats to export (`json`, `csv`, `html`)
- `output_prefix` – prefix for generated files
- `visualize` – enable image and interactive chart generation

Any command line flags override the values loaded from the configuration file.

## Usage

Run the analysis using the configuration file only:

```bash
python3 makesentsofit.py
```

You can still pass flags to override configuration values:

```bash
python3 makesentsofit.py --queries "test" --time 7 --visualize
```

The results are written to the directory specified by `output_directory` in the
configuration.

## Running the Test Suite

Use `pytest` to run the unit tests:

```bash
pytest
```

## Limitations

Twitter scraping doesn't seem to be working, but reddit scraping is - I'll fix twitter scraping down the line
