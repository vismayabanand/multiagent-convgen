"""
Fetch a representative subset of ToolBench tool definitions.

ToolBench's raw API data is in the toolenv/tools/ directory on HuggingFace.
Each category is a subdirectory; each tool is a JSON file.

We fetch via the HuggingFace Hub API (no auth required for public datasets).
Target: ~500-1000 tools across ~10 categories for a meaningful graph.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# HuggingFace dataset API endpoint
HF_API = "https://huggingface.co/datasets/qiantong-xu/toolbench-data/resolve/main"
HF_TREE = "https://huggingface.co/api/datasets/qiantong-xu/toolbench-data/tree/main"

# Alternative: direct GitHub mirror of ToolBench tools
# The original ToolBench repo has the raw tools in a specific structure
TOOLBENCH_RAW = "https://github.com/OpenBMB/ToolBench/raw/master/data/toolenv/tools"

# We'll use the standardized tool JSON from a known working source
# ToolBench G1 data — individual tool JSON files
RAPIDAPI_TOOLS_BASE = (
    "https://huggingface.co/datasets/qiantong-xu/toolbench-data/resolve/main/toolenv/tools"
)

OUT_DIR = Path(__file__).parent.parent / "data" / "toolbench"


def fetch_url(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "toolgen/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise
    return None


def fetch_hf_tree(path: str = "") -> list[dict]:
    """Fetch directory listing from HuggingFace dataset tree API."""
    url = f"{HF_TREE}/{path}" if path else HF_TREE
    try:
        data = fetch_url(url)
        if data:
            return json.loads(data)
    except Exception as e:
        print(f"  Tree fetch failed for {path}: {e}")
    return []


def download_tool_file(category: str, tool_name: str, filename: str) -> bool:
    """Download a single tool JSON file."""
    out_path = OUT_DIR / category / tool_name / filename
    if out_path.exists():
        return True  # already downloaded

    url = f"{RAPIDAPI_TOOLS_BASE}/{category}/{tool_name}/{filename}"
    data = fetch_url(url)
    if data is None:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching ToolBench tool definitions → {OUT_DIR}")

    # Fetch top-level category listing
    print("Fetching category list...")
    entries = fetch_hf_tree("toolenv/tools")

    if not entries:
        print("HuggingFace API unavailable. Using fallback synthetic data.")
        generate_synthetic_fallback()
        return

    categories = [e["path"].split("/")[-1] for e in entries
                  if e.get("type") == "directory"]
    print(f"Found {len(categories)} categories: {categories[:10]}...")

    target_per_category = 8   # tools per category
    total_downloaded = 0
    max_categories = 15

    for cat in categories[:max_categories]:
        print(f"\n  Category: {cat}")
        tools_in_cat = fetch_hf_tree(f"toolenv/tools/{cat}")
        tool_dirs = [e for e in tools_in_cat if e.get("type") == "directory"]

        for tool_entry in tool_dirs[:target_per_category]:
            tool_name = tool_entry["path"].split("/")[-1]
            # Each tool dir has an api_list.json
            ok = download_tool_file(cat, tool_name, "api_list.json")
            if ok:
                total_downloaded += 1
                print(f"    ✓ {tool_name}")
            else:
                print(f"    ✗ {tool_name} (not found)")
            time.sleep(0.1)  # rate limit

    print(f"\nDownloaded {total_downloaded} tool files to {OUT_DIR}")


def generate_synthetic_fallback():
    """
    Generate a representative synthetic ToolBench dataset when the real
    dataset is unavailable. Covers 10 domains with realistic tool schemas
    that demonstrate the full pipeline including output→input chaining.
    """
    print("Generating synthetic ToolBench-format data...")

    tools_data = {
        "Travel": [
            {
                "category_name": "Travel",
                "tool_name": "booking_com",
                "api_name": "search_hotels",
                "api_description": "Search for available hotels in a city with optional filters",
                "required_parameters": [
                    {"name": "city", "type": "string", "description": "City name"},
                    {"name": "check_in_date", "type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                    {"name": "check_out_date", "type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                ],
                "optional_parameters": [
                    {"name": "max_price", "type": "number", "description": "Maximum price per night in USD"},
                    {"name": "min_rating", "type": "number", "description": "Minimum rating (1-5)"},
                    {"name": "num_guests", "type": "number", "description": "Number of guests"},
                ],
                "response": {
                    "properties": {
                        "hotel_id": {"type": "string", "description": "Unique hotel identifier"},
                        "name": {"type": "string", "description": "Hotel name"},
                        "price_per_night": {"type": "number", "description": "Price per night"},
                        "rating": {"type": "number", "description": "Hotel rating"},
                        "address": {"type": "string", "description": "Hotel address"},
                        "available_rooms": {"type": "number", "description": "Available rooms"},
                    }
                },
            },
            {
                "category_name": "Travel",
                "tool_name": "booking_com",
                "api_name": "book_hotel",
                "api_description": "Book a hotel room using a hotel ID obtained from search",
                "required_parameters": [
                    {"name": "hotel_id", "type": "string", "description": "Hotel ID from search results"},
                    {"name": "check_in_date", "type": "string", "description": "Check-in date"},
                    {"name": "check_out_date", "type": "string", "description": "Check-out date"},
                    {"name": "guest_name", "type": "string", "description": "Full name of the guest"},
                ],
                "optional_parameters": [
                    {"name": "num_rooms", "type": "number", "description": "Number of rooms to book"},
                    {"name": "special_requests", "type": "string", "description": "Special requests"},
                ],
                "response": {
                    "properties": {
                        "booking_id": {"type": "string", "description": "Unique booking reference"},
                        "status": {"type": "string", "description": "Booking status"},
                        "confirmation_code": {"type": "string", "description": "Confirmation code"},
                        "total_price": {"type": "number", "description": "Total price for the stay"},
                    }
                },
            },
            {
                "category_name": "Travel",
                "tool_name": "booking_com",
                "api_name": "get_hotel_reviews",
                "api_description": "Get customer reviews for a specific hotel",
                "required_parameters": [
                    {"name": "hotel_id", "type": "string", "description": "Hotel ID to get reviews for"},
                ],
                "optional_parameters": [
                    {"name": "limit", "type": "number", "description": "Max number of reviews to return"},
                    {"name": "sort_by", "type": "string", "description": "Sort order: recent | rating"},
                ],
                "response": {
                    "properties": {
                        "review_id": {"type": "string", "description": "Review identifier"},
                        "rating": {"type": "number", "description": "Reviewer rating"},
                        "comment": {"type": "string", "description": "Review text"},
                        "reviewer_name": {"type": "string", "description": "Name of reviewer"},
                        "date": {"type": "string", "description": "Review date"},
                    }
                },
            },
            {
                "category_name": "Travel",
                "tool_name": "skyscanner",
                "api_name": "search_flights",
                "api_description": "Search for available flights between two airports",
                "required_parameters": [
                    {"name": "origin", "type": "string", "description": "Origin airport code (e.g., JFK)"},
                    {"name": "destination", "type": "string", "description": "Destination airport code"},
                    {"name": "departure_date", "type": "string", "description": "Departure date (YYYY-MM-DD)"},
                ],
                "optional_parameters": [
                    {"name": "return_date", "type": "string", "description": "Return date for round trip"},
                    {"name": "passengers", "type": "number", "description": "Number of passengers"},
                    {"name": "cabin_class", "type": "string", "description": "economy | business | first"},
                ],
                "response": {
                    "properties": {
                        "flight_id": {"type": "string", "description": "Flight identifier"},
                        "airline": {"type": "string", "description": "Airline name"},
                        "departure_time": {"type": "string", "description": "Departure time"},
                        "arrival_time": {"type": "string", "description": "Arrival time"},
                        "price": {"type": "number", "description": "Price per passenger"},
                        "stops": {"type": "number", "description": "Number of stops"},
                    }
                },
            },
            {
                "category_name": "Travel",
                "tool_name": "skyscanner",
                "api_name": "book_flight",
                "api_description": "Book a flight using a flight ID obtained from search",
                "required_parameters": [
                    {"name": "flight_id", "type": "string", "description": "Flight ID from search results"},
                    {"name": "passenger_name", "type": "string", "description": "Full passenger name"},
                    {"name": "passport_number", "type": "string", "description": "Passport number"},
                ],
                "optional_parameters": [
                    {"name": "seat_preference", "type": "string", "description": "window | aisle | middle"},
                    {"name": "add_luggage", "type": "boolean", "description": "Add checked luggage"},
                ],
                "response": {
                    "properties": {
                        "booking_ref": {"type": "string", "description": "Booking reference code"},
                        "status": {"type": "string", "description": "Booking status"},
                        "seat_number": {"type": "string", "description": "Assigned seat number"},
                        "boarding_pass_url": {"type": "string", "description": "URL to download boarding pass"},
                    }
                },
            },
        ],
        "Weather": [
            {
                "category_name": "Weather",
                "tool_name": "openweathermap",
                "api_name": "get_current_weather",
                "api_description": "Get current weather conditions for a city",
                "required_parameters": [
                    {"name": "city", "type": "string", "description": "City name"},
                ],
                "optional_parameters": [
                    {"name": "units", "type": "string", "description": "metric | imperial | kelvin"},
                    {"name": "country_code", "type": "string", "description": "ISO country code"},
                ],
                "response": {
                    "properties": {
                        "temperature": {"type": "number", "description": "Current temperature"},
                        "feels_like": {"type": "number", "description": "Feels like temperature"},
                        "humidity": {"type": "number", "description": "Humidity percentage"},
                        "description": {"type": "string", "description": "Weather description"},
                        "wind_speed": {"type": "number", "description": "Wind speed"},
                        "visibility": {"type": "number", "description": "Visibility in meters"},
                    }
                },
            },
            {
                "category_name": "Weather",
                "tool_name": "openweathermap",
                "api_name": "get_forecast",
                "api_description": "Get 5-day weather forecast for a city",
                "required_parameters": [
                    {"name": "city", "type": "string", "description": "City name"},
                ],
                "optional_parameters": [
                    {"name": "days", "type": "number", "description": "Number of forecast days (1-5)"},
                    {"name": "units", "type": "string", "description": "metric | imperial"},
                ],
                "response": {
                    "properties": {
                        "forecast_id": {"type": "string", "description": "Forecast ID"},
                        "date": {"type": "string", "description": "Forecast date"},
                        "high_temp": {"type": "number", "description": "High temperature"},
                        "low_temp": {"type": "number", "description": "Low temperature"},
                        "precipitation_chance": {"type": "number", "description": "Chance of rain (%)"},
                        "description": {"type": "string", "description": "Forecast description"},
                    }
                },
            },
            {
                "category_name": "Weather",
                "tool_name": "weather_alerts",
                "api_name": "get_alerts",
                "api_description": "Get active weather alerts and warnings for a region",
                "required_parameters": [
                    {"name": "region", "type": "string", "description": "Region or city name"},
                ],
                "optional_parameters": [
                    {"name": "severity", "type": "string", "description": "minor | moderate | severe | extreme"},
                ],
                "response": {
                    "properties": {
                        "alert_id": {"type": "string", "description": "Alert identifier"},
                        "type": {"type": "string", "description": "Alert type (e.g., tornado watch)"},
                        "severity": {"type": "string", "description": "Alert severity"},
                        "message": {"type": "string", "description": "Alert message"},
                        "expires": {"type": "string", "description": "Alert expiration time"},
                    }
                },
            },
        ],
        "Finance": [
            {
                "category_name": "Finance",
                "tool_name": "alpha_vantage",
                "api_name": "get_stock_quote",
                "api_description": "Get real-time stock price and trading information",
                "required_parameters": [
                    {"name": "symbol", "type": "string", "description": "Stock ticker symbol (e.g., AAPL)"},
                ],
                "optional_parameters": [
                    {"name": "exchange", "type": "string", "description": "Stock exchange (NYSE, NASDAQ)"},
                ],
                "response": {
                    "properties": {
                        "stock_id": {"type": "string", "description": "Internal stock identifier"},
                        "symbol": {"type": "string", "description": "Ticker symbol"},
                        "price": {"type": "number", "description": "Current price"},
                        "change": {"type": "number", "description": "Price change today"},
                        "change_percent": {"type": "number", "description": "Percentage change"},
                        "volume": {"type": "number", "description": "Trading volume"},
                        "market_cap": {"type": "number", "description": "Market capitalization"},
                    }
                },
            },
            {
                "category_name": "Finance",
                "tool_name": "alpha_vantage",
                "api_name": "get_stock_history",
                "api_description": "Get historical price data for a stock",
                "required_parameters": [
                    {"name": "symbol", "type": "string", "description": "Stock ticker symbol"},
                    {"name": "period", "type": "string", "description": "1d | 5d | 1m | 3m | 1y"},
                ],
                "optional_parameters": [
                    {"name": "interval", "type": "string", "description": "Data interval: 1min | 5min | daily"},
                ],
                "response": {
                    "properties": {
                        "history_id": {"type": "string", "description": "History record ID"},
                        "date": {"type": "string", "description": "Date"},
                        "open": {"type": "number", "description": "Opening price"},
                        "close": {"type": "number", "description": "Closing price"},
                        "high": {"type": "number", "description": "Daily high"},
                        "low": {"type": "number", "description": "Daily low"},
                        "volume": {"type": "number", "description": "Trading volume"},
                    }
                },
            },
            {
                "category_name": "Finance",
                "tool_name": "currency_exchange",
                "api_name": "convert_currency",
                "api_description": "Convert an amount from one currency to another",
                "required_parameters": [
                    {"name": "from_currency", "type": "string", "description": "Source currency code (e.g., USD)"},
                    {"name": "to_currency", "type": "string", "description": "Target currency code"},
                    {"name": "amount", "type": "number", "description": "Amount to convert"},
                ],
                "optional_parameters": [],
                "response": {
                    "properties": {
                        "conversion_id": {"type": "string", "description": "Conversion transaction ID"},
                        "converted_amount": {"type": "number", "description": "Converted amount"},
                        "exchange_rate": {"type": "number", "description": "Exchange rate used"},
                        "timestamp": {"type": "string", "description": "Conversion timestamp"},
                    }
                },
            },
            {
                "category_name": "Finance",
                "tool_name": "portfolio_tracker",
                "api_name": "add_to_watchlist",
                "api_description": "Add a stock to the user's watchlist for tracking",
                "required_parameters": [
                    {"name": "symbol", "type": "string", "description": "Stock ticker symbol to watch"},
                    {"name": "user_id", "type": "string", "description": "User account ID"},
                ],
                "optional_parameters": [
                    {"name": "alert_price", "type": "number", "description": "Price alert threshold"},
                    {"name": "notes", "type": "string", "description": "Personal notes about this stock"},
                ],
                "response": {
                    "properties": {
                        "watchlist_id": {"type": "string", "description": "Watchlist entry ID"},
                        "status": {"type": "string", "description": "success | already_exists"},
                        "added_at": {"type": "string", "description": "Timestamp when added"},
                    }
                },
            },
        ],
        "Food": [
            {
                "category_name": "Food",
                "tool_name": "yelp",
                "api_name": "search_restaurants",
                "api_description": "Search for restaurants by location and cuisine type",
                "required_parameters": [
                    {"name": "location", "type": "string", "description": "City or neighborhood"},
                    {"name": "cuisine", "type": "string", "description": "Cuisine type (italian, sushi, etc.)"},
                ],
                "optional_parameters": [
                    {"name": "price_range", "type": "string", "description": "$ | $$ | $$$ | $$$$"},
                    {"name": "min_rating", "type": "number", "description": "Minimum Yelp rating (1-5)"},
                    {"name": "open_now", "type": "boolean", "description": "Only show currently open restaurants"},
                ],
                "response": {
                    "properties": {
                        "restaurant_id": {"type": "string", "description": "Restaurant identifier"},
                        "name": {"type": "string", "description": "Restaurant name"},
                        "rating": {"type": "number", "description": "Yelp rating"},
                        "price_range": {"type": "string", "description": "Price range indicator"},
                        "address": {"type": "string", "description": "Street address"},
                        "phone": {"type": "string", "description": "Phone number"},
                    }
                },
            },
            {
                "category_name": "Food",
                "tool_name": "opentable",
                "api_name": "book_table",
                "api_description": "Make a restaurant reservation using the restaurant ID",
                "required_parameters": [
                    {"name": "restaurant_id", "type": "string", "description": "Restaurant ID from search"},
                    {"name": "date", "type": "string", "description": "Reservation date (YYYY-MM-DD)"},
                    {"name": "time", "type": "string", "description": "Reservation time (HH:MM)"},
                    {"name": "party_size", "type": "number", "description": "Number of diners"},
                ],
                "optional_parameters": [
                    {"name": "special_occasion", "type": "string", "description": "birthday | anniversary | business"},
                    {"name": "dietary_restrictions", "type": "string", "description": "Any dietary needs"},
                ],
                "response": {
                    "properties": {
                        "reservation_id": {"type": "string", "description": "Reservation reference"},
                        "status": {"type": "string", "description": "confirmed | waitlisted"},
                        "confirmation_number": {"type": "string", "description": "Confirmation number"},
                        "table_number": {"type": "number", "description": "Reserved table number"},
                    }
                },
            },
            {
                "category_name": "Food",
                "tool_name": "yelp",
                "api_name": "get_menu",
                "api_description": "Get the menu for a specific restaurant",
                "required_parameters": [
                    {"name": "restaurant_id", "type": "string", "description": "Restaurant ID"},
                ],
                "optional_parameters": [
                    {"name": "section", "type": "string", "description": "Menu section: appetizers | mains | desserts"},
                ],
                "response": {
                    "properties": {
                        "item_id": {"type": "string", "description": "Menu item identifier"},
                        "name": {"type": "string", "description": "Item name"},
                        "description": {"type": "string", "description": "Item description"},
                        "price": {"type": "number", "description": "Item price"},
                        "category": {"type": "string", "description": "Menu category"},
                    }
                },
            },
        ],
        "Sports": [
            {
                "category_name": "Sports",
                "tool_name": "espn_api",
                "api_name": "get_scores",
                "api_description": "Get live or recent scores for a sport",
                "required_parameters": [
                    {"name": "sport", "type": "string", "description": "Sport type: nba | nfl | mlb | soccer"},
                ],
                "optional_parameters": [
                    {"name": "date", "type": "string", "description": "Date for scores (YYYY-MM-DD), defaults to today"},
                    {"name": "league", "type": "string", "description": "Specific league name"},
                ],
                "response": {
                    "properties": {
                        "game_id": {"type": "string", "description": "Game identifier"},
                        "home_team": {"type": "string", "description": "Home team name"},
                        "away_team": {"type": "string", "description": "Away team name"},
                        "home_score": {"type": "number", "description": "Home team score"},
                        "away_score": {"type": "number", "description": "Away team score"},
                        "status": {"type": "string", "description": "live | final | scheduled"},
                        "quarter": {"type": "string", "description": "Current period or quarter"},
                    }
                },
            },
            {
                "category_name": "Sports",
                "tool_name": "espn_api",
                "api_name": "get_game_details",
                "api_description": "Get detailed statistics for a specific game",
                "required_parameters": [
                    {"name": "game_id", "type": "string", "description": "Game ID from scores endpoint"},
                ],
                "optional_parameters": [
                    {"name": "include_play_by_play", "type": "boolean", "description": "Include play-by-play data"},
                ],
                "response": {
                    "properties": {
                        "detail_id": {"type": "string", "description": "Detail record ID"},
                        "top_scorer": {"type": "string", "description": "Top scoring player"},
                        "total_points": {"type": "number", "description": "Total combined points"},
                        "attendance": {"type": "number", "description": "Game attendance"},
                        "venue": {"type": "string", "description": "Stadium or arena name"},
                    }
                },
            },
            {
                "category_name": "Sports",
                "tool_name": "ticketmaster",
                "api_name": "find_sports_tickets",
                "api_description": "Find tickets for sporting events",
                "required_parameters": [
                    {"name": "event_type", "type": "string", "description": "Sport or event type"},
                    {"name": "city", "type": "string", "description": "City to find events in"},
                ],
                "optional_parameters": [
                    {"name": "max_price", "type": "number", "description": "Maximum ticket price"},
                    {"name": "date_from", "type": "string", "description": "Start date range"},
                    {"name": "date_to", "type": "string", "description": "End date range"},
                ],
                "response": {
                    "properties": {
                        "ticket_id": {"type": "string", "description": "Ticket listing ID"},
                        "event_name": {"type": "string", "description": "Event name"},
                        "date": {"type": "string", "description": "Event date"},
                        "venue": {"type": "string", "description": "Venue name"},
                        "price": {"type": "number", "description": "Ticket price"},
                        "section": {"type": "string", "description": "Seating section"},
                    }
                },
            },
        ],
        "News": [
            {
                "category_name": "News",
                "tool_name": "newsapi",
                "api_name": "search_news",
                "api_description": "Search for news articles by keyword or topic",
                "required_parameters": [
                    {"name": "query", "type": "string", "description": "Search query or topic"},
                ],
                "optional_parameters": [
                    {"name": "language", "type": "string", "description": "Language code (en, es, fr)"},
                    {"name": "from_date", "type": "string", "description": "From date (YYYY-MM-DD)"},
                    {"name": "to_date", "type": "string", "description": "To date (YYYY-MM-DD)"},
                    {"name": "sort_by", "type": "string", "description": "relevancy | popularity | publishedAt"},
                ],
                "response": {
                    "properties": {
                        "article_id": {"type": "string", "description": "Article identifier"},
                        "title": {"type": "string", "description": "Article title"},
                        "description": {"type": "string", "description": "Article summary"},
                        "source": {"type": "string", "description": "News source name"},
                        "published_at": {"type": "string", "description": "Publication timestamp"},
                        "url": {"type": "string", "description": "Article URL"},
                    }
                },
            },
            {
                "category_name": "News",
                "tool_name": "newsapi",
                "api_name": "get_article_content",
                "api_description": "Get the full text content of a news article by ID",
                "required_parameters": [
                    {"name": "article_id", "type": "string", "description": "Article ID from search results"},
                ],
                "optional_parameters": [
                    {"name": "include_related", "type": "boolean", "description": "Include related articles"},
                ],
                "response": {
                    "properties": {
                        "content_id": {"type": "string", "description": "Content record ID"},
                        "full_text": {"type": "string", "description": "Full article text"},
                        "word_count": {"type": "number", "description": "Article word count"},
                        "reading_time": {"type": "number", "description": "Estimated reading time in minutes"},
                        "author": {"type": "string", "description": "Article author"},
                    }
                },
            },
        ],
        "Shopping": [
            {
                "category_name": "Shopping",
                "tool_name": "amazon_search",
                "api_name": "search_products",
                "api_description": "Search for products on Amazon by keyword",
                "required_parameters": [
                    {"name": "keyword", "type": "string", "description": "Product search keyword"},
                ],
                "optional_parameters": [
                    {"name": "max_price", "type": "number", "description": "Maximum price"},
                    {"name": "min_rating", "type": "number", "description": "Minimum customer rating"},
                    {"name": "prime_only", "type": "boolean", "description": "Only Prime-eligible items"},
                    {"name": "category", "type": "string", "description": "Product category"},
                ],
                "response": {
                    "properties": {
                        "product_id": {"type": "string", "description": "Product ASIN or identifier"},
                        "name": {"type": "string", "description": "Product name"},
                        "price": {"type": "number", "description": "Product price"},
                        "rating": {"type": "number", "description": "Customer rating"},
                        "review_count": {"type": "number", "description": "Number of reviews"},
                        "prime_eligible": {"type": "boolean", "description": "Prime eligible"},
                    }
                },
            },
            {
                "category_name": "Shopping",
                "tool_name": "amazon_search",
                "api_name": "get_product_details",
                "api_description": "Get detailed information about a specific product",
                "required_parameters": [
                    {"name": "product_id", "type": "string", "description": "Product ID from search results"},
                ],
                "optional_parameters": [],
                "response": {
                    "properties": {
                        "detail_id": {"type": "string", "description": "Detail record ID"},
                        "description": {"type": "string", "description": "Full product description"},
                        "dimensions": {"type": "string", "description": "Product dimensions"},
                        "weight": {"type": "number", "description": "Product weight"},
                        "in_stock": {"type": "boolean", "description": "Whether item is in stock"},
                        "shipping_days": {"type": "number", "description": "Estimated shipping days"},
                    }
                },
            },
            {
                "category_name": "Shopping",
                "tool_name": "amazon_cart",
                "api_name": "add_to_cart",
                "api_description": "Add a product to the shopping cart",
                "required_parameters": [
                    {"name": "product_id", "type": "string", "description": "Product ID to add"},
                    {"name": "quantity", "type": "number", "description": "Quantity to add"},
                ],
                "optional_parameters": [
                    {"name": "size", "type": "string", "description": "Product size if applicable"},
                    {"name": "color", "type": "string", "description": "Product color if applicable"},
                ],
                "response": {
                    "properties": {
                        "cart_item_id": {"type": "string", "description": "Cart item identifier"},
                        "cart_total": {"type": "number", "description": "Current cart total"},
                        "status": {"type": "string", "description": "added | out_of_stock | limit_exceeded"},
                    }
                },
            },
        ],
        "Health": [
            {
                "category_name": "Health",
                "tool_name": "medical_api",
                "api_name": "search_symptoms",
                "api_description": "Search for information about medical symptoms",
                "required_parameters": [
                    {"name": "symptom", "type": "string", "description": "Symptom to look up"},
                ],
                "optional_parameters": [
                    {"name": "age_group", "type": "string", "description": "child | adult | elderly"},
                    {"name": "severity", "type": "string", "description": "mild | moderate | severe"},
                ],
                "response": {
                    "properties": {
                        "symptom_id": {"type": "string", "description": "Symptom record ID"},
                        "description": {"type": "string", "description": "Symptom description"},
                        "possible_causes": {"type": "string", "description": "Possible causes"},
                        "urgency_level": {"type": "string", "description": "Urgency: low | medium | high"},
                        "recommended_action": {"type": "string", "description": "Recommended course of action"},
                    }
                },
            },
            {
                "category_name": "Health",
                "tool_name": "pharmacy_api",
                "api_name": "find_medication",
                "api_description": "Look up medication information and availability",
                "required_parameters": [
                    {"name": "medication_name", "type": "string", "description": "Name of medication"},
                ],
                "optional_parameters": [
                    {"name": "generic_ok", "type": "boolean", "description": "Accept generic alternatives"},
                    {"name": "pharmacy_zip", "type": "string", "description": "Zip code to find nearby pharmacies"},
                ],
                "response": {
                    "properties": {
                        "medication_id": {"type": "string", "description": "Medication identifier"},
                        "brand_name": {"type": "string", "description": "Brand name"},
                        "generic_name": {"type": "string", "description": "Generic name"},
                        "dosage_forms": {"type": "string", "description": "Available dosage forms"},
                        "requires_prescription": {"type": "boolean", "description": "Prescription required"},
                        "average_price": {"type": "number", "description": "Average retail price"},
                    }
                },
            },
        ],
        "Maps": [
            {
                "category_name": "Maps",
                "tool_name": "google_maps",
                "api_name": "get_directions",
                "api_description": "Get driving or transit directions between two locations",
                "required_parameters": [
                    {"name": "origin", "type": "string", "description": "Starting address or location"},
                    {"name": "destination", "type": "string", "description": "Destination address or location"},
                ],
                "optional_parameters": [
                    {"name": "mode", "type": "string", "description": "driving | walking | transit | cycling"},
                    {"name": "avoid", "type": "string", "description": "tolls | highways | ferries"},
                ],
                "response": {
                    "properties": {
                        "route_id": {"type": "string", "description": "Route identifier"},
                        "distance_km": {"type": "number", "description": "Total distance in km"},
                        "duration_minutes": {"type": "number", "description": "Estimated travel time"},
                        "steps": {"type": "string", "description": "Turn-by-turn directions"},
                        "toll_cost": {"type": "number", "description": "Estimated toll cost"},
                    }
                },
            },
            {
                "category_name": "Maps",
                "tool_name": "google_maps",
                "api_name": "search_nearby",
                "api_description": "Find nearby places of interest (restaurants, gas stations, etc.)",
                "required_parameters": [
                    {"name": "location", "type": "string", "description": "Center location for search"},
                    {"name": "place_type", "type": "string", "description": "Type: restaurant | gas_station | hospital | etc"},
                ],
                "optional_parameters": [
                    {"name": "radius_km", "type": "number", "description": "Search radius in km"},
                    {"name": "min_rating", "type": "number", "description": "Minimum Google rating"},
                ],
                "response": {
                    "properties": {
                        "place_id": {"type": "string", "description": "Place identifier"},
                        "name": {"type": "string", "description": "Place name"},
                        "address": {"type": "string", "description": "Full address"},
                        "rating": {"type": "number", "description": "Google rating"},
                        "distance_km": {"type": "number", "description": "Distance from search center"},
                        "open_now": {"type": "boolean", "description": "Currently open"},
                    }
                },
            },
            {
                "category_name": "Maps",
                "tool_name": "parking_api",
                "api_name": "find_parking",
                "api_description": "Find available parking near a location",
                "required_parameters": [
                    {"name": "location", "type": "string", "description": "Location to find parking near"},
                ],
                "optional_parameters": [
                    {"name": "max_price_per_hour", "type": "number", "description": "Maximum hourly rate"},
                    {"name": "duration_hours", "type": "number", "description": "Expected parking duration"},
                    {"name": "ev_charging", "type": "boolean", "description": "Require EV charging"},
                ],
                "response": {
                    "properties": {
                        "parking_id": {"type": "string", "description": "Parking lot identifier"},
                        "name": {"type": "string", "description": "Parking lot name"},
                        "address": {"type": "string", "description": "Parking lot address"},
                        "price_per_hour": {"type": "number", "description": "Hourly rate"},
                        "available_spots": {"type": "number", "description": "Available parking spots"},
                        "distance_meters": {"type": "number", "description": "Distance from target location"},
                    }
                },
            },
        ],
        "Productivity": [
            {
                "category_name": "Productivity",
                "tool_name": "google_calendar",
                "api_name": "create_event",
                "api_description": "Create a new calendar event",
                "required_parameters": [
                    {"name": "title", "type": "string", "description": "Event title"},
                    {"name": "start_time", "type": "string", "description": "Start datetime (ISO 8601)"},
                    {"name": "end_time", "type": "string", "description": "End datetime (ISO 8601)"},
                ],
                "optional_parameters": [
                    {"name": "description", "type": "string", "description": "Event description"},
                    {"name": "location", "type": "string", "description": "Event location"},
                    {"name": "attendees", "type": "string", "description": "Comma-separated email addresses"},
                    {"name": "reminder_minutes", "type": "number", "description": "Reminder before event in minutes"},
                ],
                "response": {
                    "properties": {
                        "event_id": {"type": "string", "description": "Calendar event identifier"},
                        "status": {"type": "string", "description": "confirmed | tentative"},
                        "calendar_link": {"type": "string", "description": "Link to view event"},
                        "created_at": {"type": "string", "description": "Creation timestamp"},
                    }
                },
            },
            {
                "category_name": "Productivity",
                "tool_name": "google_calendar",
                "api_name": "list_events",
                "api_description": "List upcoming calendar events",
                "required_parameters": [],
                "optional_parameters": [
                    {"name": "from_date", "type": "string", "description": "Start of date range"},
                    {"name": "to_date", "type": "string", "description": "End of date range"},
                    {"name": "max_results", "type": "number", "description": "Maximum events to return"},
                ],
                "response": {
                    "properties": {
                        "event_id": {"type": "string", "description": "Event identifier"},
                        "title": {"type": "string", "description": "Event title"},
                        "start_time": {"type": "string", "description": "Start time"},
                        "end_time": {"type": "string", "description": "End time"},
                        "location": {"type": "string", "description": "Event location"},
                    }
                },
            },
            {
                "category_name": "Productivity",
                "tool_name": "task_manager",
                "api_name": "create_task",
                "api_description": "Create a new task or to-do item",
                "required_parameters": [
                    {"name": "title", "type": "string", "description": "Task title"},
                ],
                "optional_parameters": [
                    {"name": "due_date", "type": "string", "description": "Due date (YYYY-MM-DD)"},
                    {"name": "priority", "type": "string", "description": "low | medium | high | urgent"},
                    {"name": "project", "type": "string", "description": "Project name to assign task to"},
                    {"name": "notes", "type": "string", "description": "Additional task notes"},
                ],
                "response": {
                    "properties": {
                        "task_id": {"type": "string", "description": "Task identifier"},
                        "status": {"type": "string", "description": "Task status: pending"},
                        "created_at": {"type": "string", "description": "Creation timestamp"},
                        "url": {"type": "string", "description": "Link to view task"},
                    }
                },
            },
        ],
    }

    total = 0
    for category, tools in tools_data.items():
        cat_dir = OUT_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        out_file = cat_dir / "api_list.json"
        with open(out_file, "w") as f:
            json.dump(tools, f, indent=2)
        total += len(tools)
        print(f"  {category}: {len(tools)} tools")

    print(f"\nGenerated {total} synthetic tools across {len(tools_data)} categories")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
