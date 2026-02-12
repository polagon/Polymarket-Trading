"""
Fetches NOAA weather forecast data (free, no API key needed).
Parses weather market questions and estimates probabilities from official forecasts.
"""
import re
import asyncio
from dataclasses import dataclass
from typing import Optional
import aiohttp

from config import NOAA_API_URL


# Common city/location coordinates for weather markets
LOCATION_MAP = {
    "new york": (40.7128, -74.0060),
    "nyc": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "la": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "miami": (25.7617, -80.1918),
    "dallas": (32.7767, -96.7970),
    "atlanta": (33.7490, -84.3880),
    "seattle": (47.6062, -122.3321),
    "san francisco": (37.7749, -122.4194),
    "denver": (39.7392, -104.9903),
    "phoenix": (33.4484, -112.0740),
    "boston": (42.3601, -71.0589),
    "washington dc": (38.9072, -77.0369),
    "philadelphia": (39.9526, -75.1652),
    "minneapolis": (44.9778, -93.2650),
    "new orleans": (29.9511, -90.0715),
    "las vegas": (36.1699, -115.1398),
    "portland": (45.5231, -122.6765),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "tokyo": (35.6762, 139.6503),
}


@dataclass
class NOAAForecast:
    location: str
    temperature_high_f: Optional[float]
    temperature_low_f: Optional[float]
    precipitation_chance: Optional[float]  # 0-100
    short_forecast: str
    detailed_forecast: str


async def _get_grid_point(session: aiohttp.ClientSession, lat: float, lon: float) -> Optional[dict]:
    url = f"{NOAA_API_URL}/points/{lat:.4f},{lon:.4f}"
    headers = {"User-Agent": "PolymarketScanner/1.0 (educational research)"}
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return None


async def _get_forecast(session: aiohttp.ClientSession, forecast_url: str) -> Optional[dict]:
    headers = {"User-Agent": "PolymarketScanner/1.0 (educational research)"}
    try:
        async with session.get(forecast_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return None


async def fetch_forecast(location: str) -> Optional[NOAAForecast]:
    """
    Fetch NOAA forecast for a named location.
    Returns None if location not found or API unavailable.
    """
    loc_key = location.lower().strip()
    coords = None

    for name, c in LOCATION_MAP.items():
        if name in loc_key or loc_key in name:
            coords = c
            break

    if not coords:
        return None

    lat, lon = coords

    async with aiohttp.ClientSession() as session:
        grid_data = await _get_grid_point(session, lat, lon)
        if not grid_data:
            return None

        try:
            props = grid_data["properties"]
            forecast_url = props["forecast"]
        except (KeyError, TypeError):
            return None

        forecast_data = await _get_forecast(session, forecast_url)
        if not forecast_data:
            return None

        try:
            periods = forecast_data["properties"]["periods"]
            if not periods:
                return None

            # First period = today's daytime forecast
            period = periods[0]
            short = period.get("shortForecast", "")
            detailed = period.get("detailedForecast", "")

            temp = period.get("temperature")
            if period.get("temperatureUnit") == "C" and temp is not None:
                temp = temp * 9 / 5 + 32

            # Extract precipitation probability
            precip_chance = None
            if "probabilityOfPrecipitation" in period:
                pval = period["probabilityOfPrecipitation"].get("value")
                if pval is not None:
                    precip_chance = float(pval)
            else:
                # Parse from text: "Chance of rain 40%"
                m = re.search(r'(\d+)\s*%\s*chance', detailed.lower())
                if m:
                    precip_chance = float(m.group(1))

            return NOAAForecast(
                location=location,
                temperature_high_f=float(temp) if temp is not None else None,
                temperature_low_f=None,  # Need overnight period for low
                precipitation_chance=precip_chance,
                short_forecast=short,
                detailed_forecast=detailed,
            )
        except (KeyError, TypeError, ValueError):
            return None


def parse_weather_question(question: str) -> Optional[dict]:
    """
    Parse a weather market question.

    Supported patterns:
    - "Will it rain in Miami on March 1?"
    - "Will the high temperature in Chicago exceed 90°F on July 4?"
    - "Will there be a hurricane in Florida in 2025?"
    - "Will New York get more than 2 inches of snow?"
    """
    q = question.lower()

    # Find location
    location = None
    for name in LOCATION_MAP:
        if name in q:
            location = name
            break

    if not location:
        # Try to extract any capitalized city name
        m = re.search(r'in ([A-Z][a-z]+(?: [A-Z][a-z]+)?)', question)
        if m:
            location = m.group(1).lower()

    # Determine weather type
    weather_type = None
    if any(kw in q for kw in ["rain", "precipitation", "shower", "drizzle"]):
        weather_type = "precipitation"
    elif any(kw in q for kw in ["snow", "blizzard", "snowfall", "inches of snow"]):
        weather_type = "snow"
    elif any(kw in q for kw in ["temperature", "degrees", "°f", "°c", "high", "low", "exceed", "above", "below"]):
        weather_type = "temperature"
    elif any(kw in q for kw in ["hurricane", "tropical storm", "typhoon", "cyclone"]):
        weather_type = "extreme_storm"
    elif any(kw in q for kw in ["tornado", "twister"]):
        weather_type = "tornado"
    elif any(kw in q for kw in ["wildfire", "fire"]):
        weather_type = "wildfire"
    elif any(kw in q for kw in ["earthquake", "quake", "seismic"]):
        weather_type = "earthquake"

    if not weather_type:
        return None

    # For temperature questions, extract threshold
    temp_threshold = None
    if weather_type == "temperature":
        # Look for temperature value
        m = re.search(r'(\d+)\s*(?:°?\s*f|°?\s*c|degrees?)', q)
        if m:
            temp_threshold = float(m.group(1))

        direction = "above" if any(kw in q for kw in ["above", "exceed", "high", "over"]) else "below"
    else:
        direction = None
        temp_threshold = None

    return {
        "location": location,
        "weather_type": weather_type,
        "temp_threshold": temp_threshold,
        "direction": direction,
    }


def estimate_probability(question: str, forecast: Optional[NOAAForecast]) -> Optional[dict]:
    """
    Estimate probability for a weather market question given NOAA forecast data.
    """
    parsed = parse_weather_question(question)
    if not parsed:
        return None

    if not forecast:
        return None

    weather_type = parsed["weather_type"]
    probability = None
    confidence = 0.5

    if weather_type == "precipitation":
        if forecast.precipitation_chance is not None:
            probability = forecast.precipitation_chance / 100
            confidence = 0.75  # NOAA precip forecasts are well-calibrated
        else:
            # Infer from forecast text
            short = forecast.short_forecast.lower()
            if "sunny" in short or "clear" in short:
                probability = 0.05
                confidence = 0.55
            elif "partly cloudy" in short or "mostly cloudy" in short:
                probability = 0.20
                confidence = 0.45
            elif "chance" in short:
                probability = 0.35
                confidence = 0.50
            elif "likely" in short:
                probability = 0.65
                confidence = 0.55
            elif "rain" in short or "showers" in short:
                probability = 0.80
                confidence = 0.60

    elif weather_type == "temperature" and parsed["temp_threshold"] is not None:
        if forecast.temperature_high_f is not None:
            current_high = forecast.temperature_high_f
            threshold = parsed["temp_threshold"]
            diff = current_high - threshold

            # Simple sigmoid-like probability based on how close we are to threshold
            # Forecast uncertainty ~ ±5°F for 1-day, higher for multi-day
            uncertainty_f = 6.0
            import math
            z = diff / uncertainty_f
            # Probability that actual high exceeds threshold
            prob_above = 0.5 + 0.5 * math.tanh(z * 1.0)

            if parsed["direction"] == "above":
                probability = prob_above
            else:
                probability = 1.0 - prob_above

            # Confidence higher when well above/below threshold
            confidence = min(0.80, 0.45 + abs(diff) / threshold * 0.5)

    elif weather_type == "snow":
        if forecast.precipitation_chance is not None:
            short = forecast.short_forecast.lower()
            # Only count as snow if cold enough and snow mentioned
            if "snow" in short or "flurr" in short or "blizzard" in short:
                probability = forecast.precipitation_chance / 100
                confidence = 0.65
            else:
                probability = 0.03
                confidence = 0.50

    elif weather_type in ("extreme_storm", "tornado", "wildfire", "earthquake"):
        # Rare events — NOAA doesn't give direct probabilities for these
        # Use very low base rate; low confidence
        base_rates = {
            "extreme_storm": 0.05,
            "tornado": 0.02,
            "wildfire": 0.10,
            "earthquake": 0.08,
        }
        probability = base_rates.get(weather_type, 0.05)
        confidence = 0.30  # Low confidence — NOAA data not directly applicable

    if probability is None:
        return None

    return {
        "probability": round(max(0.01, min(0.99, probability)), 4),
        "confidence": round(confidence, 4),
        "source": "noaa_forecast",
        "details": {
            "location": parsed["location"],
            "weather_type": weather_type,
            "noaa_high_f": forecast.temperature_high_f,
            "noaa_precip_pct": forecast.precipitation_chance,
            "noaa_short_forecast": forecast.short_forecast,
        }
    }
