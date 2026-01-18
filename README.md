# Aimé Leon Dore Product Scraper

A comprehensive scraper for Aimé Leon Dore fashion website that extracts all products, generates image embeddings, and stores them in Supabase.

## Features

- **Complete Product Catalog**: Scrapes all products from the shop-all collection page
- **Dynamic Loading**: Handles infinite scroll pagination
- **Detailed Product Info**: Extracts title, price, description, category, sizes, and more
- **Image Embeddings**: Generates 768-dimensional embeddings using Google SigLIP model
- **Supabase Integration**: Automatically stores all data in your Supabase database
- **Error Handling**: Robust error handling and retry mechanisms
- **Logging**: Comprehensive logging for monitoring and debugging

## Requirements

- Python 3.8+
- Chrome browser (for Selenium)
- Supabase account with the products table created

## Installation

1. Clone or download the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Database Setup

The scraper expects a Supabase table with this structure:

```sql
create table public.products (
  id text not null,
  source text null,
  product_url text null,
  affiliate_url text null,
  image_url text not null,
  brand text null,
  title text not null,
  description text null,
  category text null,
  gender text null,
  price double precision null,
  currency text null,
  search_tsv tsvector null,
  created_at timestamp with time zone null default now(),
  metadata text null,
  size text null,
  second_hand boolean null default false,
  embedding public.vector null,
  country text null,
  compressed_image_url text null,
  tags text[] null,
  search_vector tsvector null,
  constraint products_pkey primary key (id),
  constraint products_source_product_url_key unique (source, product_url)
);
```

## Usage

### Full Scraper (Recommended)
Run the complete scraper:

```bash
python run.py
```

This will scrape all products from the catalog.

### Test Version
For testing with just a few products:

```bash
python test_scraper.py
```

### Manual Run
You can also import and run the scraper directly:

```python
from scraper import AimeLeonDoreScraper
import asyncio

async def run():
    scraper = AimeLeonDoreScraper()
    product_urls = await scraper.scrape_product_urls()
    await scraper.process_products(product_urls)

asyncio.run(run())
```

## How It Works

The scraper follows these steps:

1. **Load Product Catalog**: Uses Selenium to load the shop-all collection page and scroll through all products to collect URLs
2. **Extract Product Details**: Visits each product page to extract title, price, description, images, and other metadata
3. **Generate Embeddings**: Downloads product images and generates 768-dimensional embeddings using Google SigLIP model
4. **Store in Database**: Saves all product data including embeddings to your Supabase database

## Configuration

The scraper is configured with these constants:

- `BASE_URL`: Main website URL
- `SHOP_ALL_URL`: URL for the complete product catalog
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_ANON_KEY`: Your Supabase anon key
- `EMBEDDING_MODEL_NAME`: Hugging Face model for embeddings

## Data Fields

The scraper extracts and stores:

- **id**: Unique identifier (aime-{product-slug})
- **source**: Set to "scraper"
- **product_url**: Full URL to the product page
- **image_url**: URL to the main product image
- **brand**: Set to "Aime"
- **title**: Product name
- **description**: Product description
- **category**: Product category
- **gender**: Set to "man"
- **price**: Numeric price value
- **currency**: Set to "USD"
- **second_hand**: Set to false
- **metadata**: JSON string with additional info
- **size**: Available sizes
- **country**: Set to "US"
- **embedding**: 768-dimensional vector from SigLIP model

## Logging

Logs are saved to `scraper.log` with daily rotation. Check this file for detailed information about the scraping process, errors, and progress.

## Notes

- The scraper uses Selenium to handle dynamic content loading
- Image embeddings are generated using the google/siglip-base-patch16-384 model
- The scraper includes delays and random headers to be respectful to the website
- Failed products are logged but don't stop the entire process
- Duplicate products are handled via upsert (update if exists, insert if not)

## Troubleshooting

- **Chrome Driver Issues**: Make sure Chrome is installed and up to date
- **Memory Issues**: Large catalogs might require significant RAM for embeddings
- **Rate Limiting**: The scraper includes delays but may still hit rate limits
- **Database Errors**: Check your Supabase connection and table structure

## Legal Note

This scraper is for educational and personal use only. Respect website terms of service and robots.txt. Consider the impact on the website's servers and implement appropriate delays.