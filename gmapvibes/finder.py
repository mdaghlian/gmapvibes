import requests
import time
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from urllib.parse import quote
from ddgs import DDGS
from bs4 import BeautifulSoup
from typing import Optional, List, Tuple
import concurrent.futures
import trafilatura
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SemanticURLMatcher:
    """
    A class to perform semantic similarity search on URL segments
    using sentence embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the matcher with a pre-trained model.
        
        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded successfully! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def preprocess_url_segment(self, url_segment: str) -> str:
        """
        Preprocess URL segments by replacing hyphens/underscores with spaces.
        
        Args:
            url_segment: Raw URL segment (e.g., 'sushi-rolls')
            
        Returns:
            Processed string (e.g., 'sushi rolls')
        """
        return url_segment.replace('-', ' ').replace('_', ' ')
    
    def find_similar(
        self, 
        target_string: str, 
        comparison_list: List[str], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find the top-k most semantically similar strings to the target.
        
        Args:
            target_string: The target URL segment to match against
            comparison_list: List of URL segments to compare
            top_k: Number of top results to return (default: 5)
            
        Returns:
            List of tuples (url_segment, similarity_score) sorted by similarity
        """
        # Step 1: Preprocess all strings
        processed_target = self.preprocess_url_segment(target_string)
        processed_comparisons = [self.preprocess_url_segment(s) for s in comparison_list]
        
        # Step 2: Encode target string
        print(f"\nEncoding target: '{target_string}' → '{processed_target}'")
        target_embedding = self.model.encode([processed_target], convert_to_numpy=True)
        
        # Step 3: Encode all comparison strings
        print(f"Encoding {len(comparison_list)} comparison strings...")
        comparison_embeddings = self.model.encode(processed_comparisons, convert_to_numpy=True)
        
        # Step 4: Calculate cosine similarity
        print("Calculating cosine similarities...")
        similarities = cosine_similarity(target_embedding, comparison_embeddings)[0]
        
        # Step 5: Sort and get top-k results
        # Create pairs of (original_string, similarity_score)
        results = list(zip(comparison_list, similarities))
        
        # Sort by similarity score in descending order
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return results_sorted[:top_k]
    
    def print_results(self, target: str, results: List[Tuple[str, float]]):
        """
        Pretty print the similarity search results.
        
        Args:
            target: The target string that was searched
            results: List of (string, score) tuples
        """
        print(f"\n{'='*70}")
        print(f"Target: '{target}'")
        print(f"{'='*70}")
        print(f"{'Rank':<6} {'URL Segment':<35} {'Similarity Score':<15}")
        print(f"{'-'*70}")
        
        for rank, (url_segment, score) in enumerate(results, 1):
            print(f"{rank:<6} {url_segment:<35} {score:.4f}")
        print(f"{'='*70}\n")


class OSMVibeFinder:
    """
    Find places with similar vibes using OpenStreetMap and semantic similarity.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.geolocator = Nominatim(user_agent="OSMVibes")
        self.places = []
        self.centre_info = {}
        self.target_info = {}
        self.semantic_matcher = SemanticURLMatcher(model_name=model_name)
        
    def geocode_postcode(self, postcode):
        """Convert postcode to latitude/longitude using free Nominatim."""
        print(f"🔍 Geocoding postcode: {postcode}")
        try:
            location = self.geolocator.geocode(postcode)
            if not location:
                print("❌ Could not geocode postcode")
                return None
            
            lat, lng = location.latitude, location.longitude
            print(f"✅ Found coordinates: {lat}, {lng}")
            return (lat, lng)
        except Exception as e:
            print(f"❌ Geocoding error: {e}")
            return None
        
    def _format_osm_address(self, tags):
        """Format address from OSM tags."""
        parts = []
        if tags.get('addr:housenumber'):
            parts.append(tags['addr:housenumber'])
        if tags.get('addr:street'):
            parts.append(tags['addr:street'])
        if tags.get('addr:city'):
            parts.append(tags['addr:city'])
        if tags.get('addr:postcode'):
            parts.append(tags['addr:postcode'])
        
        return ', '.join(parts) if parts else 'N/A'
    
    def set_target_vibe(self, target_url, extra_keys=[]):
        """Set the target vibe by scraping content from a URL."""
        print(f"🎯 Setting target vibe from: {target_url}")
        url_string = scrape_text_from_url(target_url)
        string_extra = ', '.join(extra_keys)
        self.target_info = {
            'url': target_url, 
            'url_string':  string_extra + url_string 
        }
        print(f"✅ Target vibe set ({len(url_string)} characters scraped)")
    
    def _get_website_for_place(self, place):
        """Find website for a place using DuckDuckGo if not available."""
        if place['website'] == 'N/A':
            try:
                results = DDGS().text(
                    place['name'] + ' ' + self.centre_info['city'], 
                    region=self.centre_info['country_code'], 
                    max_results=1
                )
                if results:
                    place['website'] = results[0]['href']
            except Exception as e:
                print(results)
                print(f"⚠️  Could not find website for {place['name']}: {e}")
        return place

    def _add_websites_concurrently(self, places):
        """Processes a list of places concurrently using a thread pool."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self._get_website_for_place, place)
                for place in places
            ]
            concurrent.futures.wait(futures)
        return places

    def _get_string_for_url(self, place):
        """Scrape text content from place's website."""
        if place['website'] != 'N/A':
            place['url_string'] = scrape_text_from_url(place['website'])
        else:
            place['url_string'] = ''
        return place

    def _add_strings_concurrently(self, places, max_workers=20):
        """Get text content for all places concurrently."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._get_string_for_url, place): place 
                for place in places
            }
            
            for future in concurrent.futures.as_completed(futures, timeout=60):
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠️  Error processing place: {e}")
        
        return places

    def calculate_similarity_scores(self, places=None):
        """
        Calculate semantic similarity between target vibe and all places.
        
        Args:
            places: List of places to score (uses self.places if None)
            
        Returns:
            List of places sorted by similarity score (highest first)
        """
        if places is None:
            places = self.places
            
        if not self.target_info.get('url_string'):
            print("❌ No target vibe set. Use set_target_vibe() first.")
            return places
        
        print(f"\n🧠 Calculating semantic similarity for {len(places)} places...")
        
        # Filter places with valid content
        valid_places = [p for p in places if p.get('url_string') and len(p['url_string']) > 50]
        print(f"📊 {len(valid_places)} places have sufficient content for comparison")
        
        if not valid_places:
            print("❌ No places with sufficient content found.")
            return places
        
        # Encode target
        target_embedding = self.semantic_matcher.model.encode(
            [self.target_info['url_string']], 
            convert_to_numpy=True
        )
        
        # Encode all place strings
        place_strings = [p['url_string'] for p in valid_places]
        place_embeddings = self.semantic_matcher.model.encode(
            place_strings, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Calculate similarities
        similarities = cosine_similarity(target_embedding, place_embeddings)[0]
        
        # Add similarity scores to places
        for place, score in zip(valid_places, similarities):
            place['similarity_score'] = float(score)
        
        # Mark places without scores
        for place in places:
            if 'similarity_score' not in place:
                place['similarity_score'] = 0.0
        
        # Sort by similarity
        places_sorted = sorted(places, key=lambda x: x['similarity_score'], reverse=True)
        
        print(f"✅ Similarity scores calculated!")
        return places_sorted

    def search_nearby(self, postcode, radius_km=2):
        """
        Main search function: finds nearby places and optionally ranks by vibe similarity.
        
        Args:
            postcode: Starting postcode/zip code
            radius_km: Search radius in kilometers
        """
        print("=" * 70)
        print("🍽️  VIBE FINDER")
        print("=" * 70)
        
        # Step 1: Geocode
        coords = self.geocode_postcode(postcode)
        if not coords:
            return []
        
        lat, lng = coords
        # Get place info
        location = self.geolocator.reverse(coords, exactly_one=True, timeout=10)
        
        if location and location.raw.get('address'):
            country_code = location.raw['address'].get('country_code', 'N/A').upper()
            print(f"Country Code: {country_code}")
        else:
            print("Could not find a location for the given coordinates.") 
            country_code = ''

        self.centre_info.update({
            'country_code': country_code,
            'city': location.raw['address'].get('city', 'N/A'), 
            'location': location, 
            'postcode': postcode, 
            'lat': lat, 
            'lng': lng, 
            'coords': coords, 
        })
        
        # Step 2: Get places from OSM
        self.places = self.get_osm_places(lat, lng, radius_km)
        
        if not self.places:
            print("\n❌ No places found.")
            return []
        
        # Step 3: Calculate similarity if target vibe is set
        if self.target_info.get('url_string'):
            self.places = self.calculate_similarity_scores(self.places)
        
        # Step 4: Display results
        self.display_results(self.places)
        
        return self.places

    def get_osm_places(self, lat, lng, radius_km):
        """
        Get places from OpenStreetMap using Overpass API.
        """
        print(f"\n🗺️  Searching OpenStreetMap within {radius_km}km...")
        
        overpass_url = "http://overpass-api.de/api/interpreter"
        radius_meters = int(radius_km * 1000)
        
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["amenity"~"^(cafe|restaurant|bar)$"](around:{radius_meters},{lat},{lng});
          way["amenity"~"^(cafe|restaurant|bar)$"](around:{radius_meters},{lat},{lng});
        );
        out body center;
        """
        
        if True:
            response = requests.get(overpass_url, params={'data': overpass_query}, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            places = []
            for element in data.get('elements', []):
                tags = element.get('tags', {})
                
                # Get coordinates
                if element['type'] == 'node':
                    place_lat, place_lng = element['lat'], element['lon']
                else:  # way
                    center = element.get('center', {})
                    place_lat, place_lng = center.get('lat'), center.get('lon')
                
                if place_lat and place_lng:
                    distance = geodesic((lat, lng), (place_lat, place_lng)).km
                    city = tags.get('addr:city', 'N/A')
                    place_info = {
                        'osm_id': element.get('id'),
                        'name': tags.get('name', 'Unnamed'),
                        'category': tags.get('amenity', 'unknown'),
                        'address': self._format_osm_address(tags),
                        'city': city, 
                        'cuisine': tags.get('cuisine', 'N/A'),
                        'phone': tags.get('phone', 'N/A'),
                        'website': tags.get('website', 'N/A'),
                        'opening_hours': tags.get('opening_hours', 'N/A'),
                        'distance_km': round(distance, 2),
                        'coordinates': (place_lat, place_lng),
                        'source': 'OSM'
                    }
                    places.append(place_info)
            
            # Enrich with websites and scraped content
            places = self._add_websites_concurrently(places)
            places = self._add_strings_concurrently(places)
            
            # Sort by distance initially
            places.sort(key=lambda x: x['distance_km'])
            print(f"✅ Found {len(places)} places on OpenStreetMap")
            return places
            
        # except Exception as e:
        #     print(f"❌ Overpass API error: {e}")
        #     return []

    def display_results(self, places):
        """Display formatted results with similarity scores if available."""
        print("\n" + "=" * 70)
        print(f"📋 RESULTS: {len(places)} places found")
        print("=" * 70)
        
        has_scores = any(p.get('similarity_score', 0) > 0 for p in places)
        
        for i, place in enumerate(places, 1):
            print(f"\n{i}. {place['name']} ({place['category'].capitalize()})")
            print(f"   📍 Distance: {place['distance_km']} km")
            
            if has_scores and place.get('similarity_score'):
                print(f"   🎯 Vibe Match: {place['similarity_score']:.4f}")
            
            print(f"   📫 Address: {place['address']}")
       
            if place['cuisine'] != 'N/A':
                print(f"   🍴 Cuisine: {place['cuisine']}")
            
            if place['phone'] != 'N/A':
                print(f"   📞 Phone: {place['phone']}")
            
            if place['website'] != 'N/A':
                print(f"   🌐 Website: {place['website']}")
            
            if place['opening_hours'] != 'N/A':
                print(f"   🕒 Hours: {place['opening_hours']}")


def scrape_text_from_url(url: str, timeout: int = 3) -> str:
    """Extract main content from URL quickly."""
    try:
        response = requests.get(
            url, 
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0'},
            allow_redirects=True
        )
        response.raise_for_status()
        
        # Fast extraction with minimal processing
        text = trafilatura.extract(
            response.content,
            include_comments=False,
            include_tables=False,
            no_fallback=True,
            favor_precision=False,
            favor_recall=True
        )
        return text if text else ''
    except requests.Timeout:
        print(f"⏱️  Timeout: {url}")
        return ''
    except Exception as e:
        return ''
