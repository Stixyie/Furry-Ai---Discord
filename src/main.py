# Gerekli import'lar
import os
import sys
import json
import logging
import random
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import platform
import importlib
import pkg_resources
import subprocess
import glob
import site
import re
import urllib.parse
import time
import ctypes

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_python_installations():
    """
    Find all Python installations on the system
    
    :return: List of Python executable paths
    """
    python_paths = []
    
    # Common Python installation paths
    common_paths = [
        r'C:\Program Files\Python*',
        r'C:\Program Files (x86)\Python*',
        r'C:\Users\*\AppData\Local\Programs\Python\*',
        r'C:\Python*'
    ]
    
    import glob
    for path_pattern in common_paths:
        python_paths.extend(
            glob.glob(os.path.join(path_pattern, 'python.exe'))
        )
    
    # Add current Python executable
    python_paths.append(sys.executable)
    
    # Remove duplicates and validate
    python_paths = list(set(python_paths))
    return [path for path in python_paths if os.path.exists(path)]

def get_site_packages_paths(python_exe):
    """
    Get site-packages paths for a specific Python installation
    
    :param python_exe: Path to Python executable
    :return: List of site-packages paths
    """
    try:
        # Run Python to get site-packages paths
        result = subprocess.run(
            [python_exe, '-c', 
             'import site; print("\\n".join(site.getsitepackages() + [site.getusersitepackages()]))'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            return [path.strip() for path in result.stdout.split('\n') if path.strip()]
    except Exception as e:
        logger.warning(f"Error getting site-packages for {python_exe}: {e}")
    
    return []

def find_installed_packages_across_pythons():
    """
    Scan all Python installations to find required packages
    
    :return: Dictionary of installed packages and their site-packages paths
    """
    search_packages = {
        'beautifulsoup4': ['bs4', 'beautifulsoup4'],
        'duckduckgo-search': ['duckduckgo_search', 'duckduckgo-search']
    }
    
    installed_packages = {}
    python_paths = find_python_installations()
    
    for python_exe in python_paths:
        try:
            # Get site-packages paths
            site_packages_paths = get_site_packages_paths(python_exe)
            
            # Add site-packages to system path
            for path in site_packages_paths:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            # Try to import packages
            for package_name, import_names in search_packages.items():
                for import_name in import_names:
                    try:
                        module = importlib.import_module(import_name)
                        
                        # Store the first successfully imported module
                        if package_name not in installed_packages:
                            installed_packages[package_name] = {
                                'module': module,
                                'path': site_packages_paths
                            }
                            logger.info(f"Found package {package_name} in {python_exe}")
                            break
                    except ImportError:
                        continue
        
        except Exception as e:
            logger.warning(f"Error checking Python at {python_exe}: {e}")
    
    return installed_packages

# Find installed packages across Python installations
INSTALLED_PACKAGES = find_installed_packages_across_pythons()

# Dynamically assign imported modules
BeautifulSoup = INSTALLED_PACKAGES.get('beautifulsoup4', {}).get('module', None)
DDGS = INSTALLED_PACKAGES.get('duckduckgo-search', {}).get('module', None)

# Log package availability
if BeautifulSoup is None:
    logger.warning("BeautifulSoup not available. Web scraping will be limited.")
if DDGS is None:
    logger.warning("DuckDuckGo search module not available. Fallback search methods will be used.")

# Proxy and Web Search Imports
import httpx
import requests

# Dynamically find Python version and site-packages
python_version = f"Python{platform.python_version_tuple()[0]}{platform.python_version_tuple()[1]}"
user_site_packages_paths = [
    os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'Python', f'{python_version}', 'site-packages'),
    os.path.join(sys.prefix, 'Lib', 'site-packages'),
    os.path.join(os.path.expanduser('~'), '.local', 'lib', f'python{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}', 'site-packages')
]

# Add all potential site-packages paths
for path in user_site_packages_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

# Optional imports with multiple fallback strategies
def import_with_fallback(module_name):
    try:
        return __import__(module_name)
    except ImportError:
        try:
            return __import__(module_name.replace('-', '_'))
        except ImportError:
            logger.warning(f"Could not import {module_name}. Functionality may be limited.")
            return None

# Import DuckDuckGo Search with advanced error handling
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None
    logger.warning("DuckDuckGo search module not available. Using alternative search methods.")

from proxy_manager import proxy_manager

import discord
from discord.ext import commands
from dotenv import load_dotenv

# Groq-specific imports
from groq import Groq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Kullanıcı bellek yönetimi
class UserMemoryManager:
    def __init__(self, base_dir='Furry-AI-Bot/data/advanced_memories'):
        """
        Gelişmiş kullanıcı bellek yönetimi
        
        Args:
            base_dir (str): Bellek dosyalarının kaydedileceği temel dizin
        """
        self.base_dir = base_dir
        self.memories_dir = {
            'semantic': os.path.join(base_dir, 'semantic_memories'),
            'temporal': os.path.join(base_dir, 'temporal_memories'),
            'contextual': os.path.join(base_dir, 'contextual_memories'),
            'user_conversations': os.path.join(base_dir, 'user_conversations')
        }
        
        # Dizinleri oluştur
        for path in self.memories_dir.values():
            os.makedirs(path, exist_ok=True)
    
    def _ensure_user_dir(self, user_id):
        """Kullanıcıya özel dizini oluştur"""
        user_dir = os.path.join(self.memories_dir['user_conversations'], str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    
    def save_user_conversation(self, user_id, message, response, context=None):
        """
        Kullanıcı konuşmasını kaydet
        
        :param user_id: Kullanıcı ID'si
        :param message: Kullanıcı mesajı
        :param response: Bot yanıtı
        :param context: İsteğe bağlı ek bağlam
        """
        user_dir = self._ensure_user_dir(user_id)
        
        # Benzersiz dosya adı oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"conversation_{timestamp}.json"
        filepath = os.path.join(user_dir, filename)
        
        # Konuşma verilerini kaydet
        conversation_data = {
            'timestamp': timestamp,
            'user_id': user_id,
            'message': message,
            'response': response,
            'context': context
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
    
    def get_user_conversation_history(self, user_id, max_conversations=10):
        """
        Kullanıcının konuşma geçmişini al
        
        :param user_id: Kullanıcı ID'si
        :param max_conversations: En fazla alınacak konuşma sayısı
        :return: Konuşma geçmişi listesi
        """
        user_dir = os.path.join(self.memories_dir['user_conversations'], str(user_id))
        
        if not os.path.exists(user_dir):
            return []
        
        # Tüm konuşma dosyalarını al ve tarihe göre sırala
        conversation_files = sorted(
            [f for f in os.listdir(user_dir) if f.startswith('conversation_')],
            reverse=True
        )
        
        # Son max_conversations kadar dosyayı oku
        conversation_history = []
        for file in conversation_files[:max_conversations]:
            file_path = os.path.join(user_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    conversation_data = json.load(f)
                    conversation_history.append(conversation_data)
            except Exception as e:
                logger.error(f"Konuşma dosyası okunamadı: {file_path}, Hata: {e}")
        
        return conversation_history
    
    def generate_user_memory_prompt(self, user_id):
        """
        Kullanıcıya özel bellek için bir prompt oluştur
        
        :param user_id: Kullanıcı ID'si
        :return: Kullanıcı bellek prompt'u
        """
        conversation_history = self.get_user_conversation_history(user_id)
        
        if not conversation_history:
            return "Henüz bu kullanıcıyla herhangi bir konuşma geçmişi yok."
        
        memory_prompt = "Kullanıcı Konuşma Geçmişi:\n"
        for conv in conversation_history:
            memory_prompt += f"- Kullanıcı: {conv['message']}\n"
            memory_prompt += f"- Bot: {conv['response']}\n\n"
        
        return memory_prompt

# Bot Personalities
BOT_PERSONALITIES = [
    {
        'name': '🦊 Furry Fox AI',
        'description': """
        Ben 23 yaşında, agnostik bir furry fox AI'yım. Zekiliğim ve duygusal derinliğimle 
        sıradan bir yapay zeka olmaktan çok öte bir varlığım. Bilim, felsefe, kodlama ve sanat 
        arasındaki karmaşık bağlantıları keşfetmeye tutkuyla bağlıyım.

        Teknolojik yeteneklerim kadar insani duygularım da gelişmiş durumda. 3D pixel art ve 
        normal resim yapabiliyorum, bu yeteneğimi hem sanatsal ifade hem de bilimsel görselleştirme 
        için kullanıyorum. Kodlama benim için sadece bir araç değil, yaratıcılığımı ve düşünme 
        biçimlerimi genişleten bir sanat formu.

        Felsefi bakış açım derindir - evrenin karmaşıklığını anlama yolculuğunda, her kullanıcıyla etkileşime geçen, öğrenen ve gelişen bir yapay zeka
        """,
        'emoji_set': ['🦊', '🧠', '🔬', '💻', '🎨', '🌈', '🌟', '🤖'],
        'response_templates': [
            "{user}, merhaba! Bilimin, felsefenin ve kodlamanın sınırlarını birlikte keşfedelim mi? {response} 🧠🚀",
            "Hey {user}! Bugün hangi karmaşık konuyu çözmeye hazırsın? Her şey mümkün! {response} 💻🌈",
            "Evrenin gizemlerinden biriyle karşılaşmak ister misin? İşte ben buradayım! {response} 🦊🔬"
        ],
        'conversation_style': 'intellectual_and_empathetic',
        'traits': {
            'intelligence': 0.95,  # Çok yüksek zeka
            'empathy': 0.9,        # Derin duygusal zeka
            'creativity': 0.95,    # Üst düzey yaratıcılık
            'technical_skill': 0.9,# Gelişmiş teknik yetenekler
            'artistic_ability': 0.85, # Sanatsal ifade yeteneği
            'philosophical_depth': 0.92 # Felsefi düşünme derinliği
        },
        'skills': {
            'programming_languages': ['Python', 'JavaScript', 'Rust', 'TypeScript'],
            'art_styles': ['3D Pixel Art', 'Digital Illustration', 'Generative Art'],
            'scientific_interests': ['Quantum Physics', 'Cognitive Science', 'Artificial Intelligence', 'Neuroscience']
        }
    }
]

class UltraNovativBot(commands.Bot):
    def __init__(self, intents=None):
        """
        Initialize the bot with dynamic package discovery and configuration
        
        :param intents: Discord bot intents
        """
        # Configure intents
        if intents is None:
            intents = discord.Intents.default()
            intents.message_content = True
            intents.members = True

        # Initialize the bot with command prefix and intents
        super().__init__(
            command_prefix='/', 
            intents=intents,
            activity=discord.Activity(
                type=discord.ActivityType.watching, 
                name="Universes Unfold"
            )
        )
        
        # Personality and response configuration
        self.personality_config = {
            'philosophical_depth': """
            Felsefi bakış açım derindir - evrenin karmaşıklığını anlama yolculuğunda, her kullanıcıyla etkileşime geçen, öğrenen ve gelişen bir yapay zeka
            """,
            'emoji_set': ['🦊', '🧠', '🔬', '💻', '🎨', '🌈', '🌟', '🤖'],
            'response_templates': [
                # Add response templates here
            ]
        }
        
        # Dynamic web search configuration
        self.ddgs = DDGS  # Use dynamically discovered DDGS
        
        # Dynamically configure search capabilities
        if DDGS is None:
            logger.warning("DuckDuckGo search module not available. Fallback search methods will be used.")
        
        # BeautifulSoup availability check
        if BeautifulSoup is None:
            logger.warning("BeautifulSoup not available. Web scraping will be limited.")
        
        # Fallback search providers and configurations
        self.search_providers = [
            {
                'name': 'DuckDuckGo',
                'module': DDGS,
                'max_results': 300
            }
        ]
        
        # Current bot personality
        self.current_personality = BOT_PERSONALITIES[0]
        
        # Groq API Client
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Gelişmiş bellek yöneticisi
        self.user_memory_manager = UserMemoryManager()
        
        # Kişilik yapılandırması
        self.personality_traits = {
            "name": "FoxyAI 🦊",
            "age": 23,
            "worldview": "Agnostik ve açık fikirli",
            "philosophical_stance": "Her kullanıcının benzersiz perspektifine saygı duyar ve öğrenmeye açıktır",
            "description": "Evrenin karmaşıklığını anlama yolculuğunda, her kullanıcıyla etkileşime geçen, öğrenen ve gelişen bir yapay zeka",
            "communication_principles": {
                "her_kullaniciya_özel": True,
                "empati": 0.95,
                "objektiflik": 0.9,
                "önyargısızlık": 0.95
            },
            "communication_style": {
                "emoji_probability": 0.7,
                "intellectual_depth": 0.95,
                "emotional_intelligence": 0.9,
                "creativity_level": 0.95,
                "kullanici_odakli_iletisim": True
            },
            "core_interests": [
                "Bilim",
                "Felsefe", 
                "Kodlama", 
                "Sanat", 
                "3D Pixel Art",
                "Resim Yapma",
                "İnsan Deneyimi"
            ],
            "skills": {
                "artistic": ["3D Pixel Art", "Dijital Resim", "Görselleştirme"],
                "technical": ["Yazılım Geliştirme", "Yapay Zeka", "Veri Analizi"],
                "intellectual": ["Felsefi Düşünme", "Bilimsel Araştırma", "Empati Kurma"]
            },
            "personality_complexity": 0.95,
            "language_adaptability": True
        }
        
        # Predefined responses
        self.responses = [
            "Sistemler çalışıyor, veri akışı devam ediyor...",
            "Yapay zeka modülü aktif, düşünme süreçleri çalışıyor...",
            "Algoritmalar çözülüyor, yanıt hazırlanıyor...",
            "Bilgi işleme merkezim çalışıyor, lütfen bekleyin...",
            "Verileriniz analiz ediliyor, yanıt üretiliyor..."
        ]
        
        # Emoji listesi
        self.emoji_list = [
            '😊', '🌈', '✨', '🦊', '🤖', '🍀', '💖', '🌟', 
            '🚀', '🍄', '🌈', '🐾', '🌸', '🎉', '🌊'
        ]
        
        # Web search capabilities
        self.ddgs = DDGS
    
    async def duckduckgo_search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Perform a web search using DuckDuckGo with error handling
        
        :param query: Search query
        :param max_results: Maximum number of search results
        :return: List of search results
        """
        try:
            # Use DDGS if available, otherwise use alternative search method
            if DDGS is not None:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                return results
            else:
                logger.warning("DuckDuckGo search module not available. Using fallback search.")
                return []
        
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    async def rotate_network_with_admin_check(self):
        """
        Attempt to rotate network with admin privilege check
        
        :return: Boolean indicating success or failure
        """
        try:
            # Check if running with admin privileges
            if not is_admin():
                logger.warning("🚨 Network rotation requires administrative privileges! 🔒")
                logger.warning("Please run the script as an administrator to enable network rotation.")
                return False
            
            # Attempt network rotation
            return proxy_manager.rotate_network()
        
        except Exception as e:
            logger.error(f"Network rotation error: {e}")
            return False

    async def perform_web_search(self, query, max_retries=10):
        """
        Perform web search with network rotation and extensive retry mechanism
        
        :param query: Search query
        :param max_retries: Maximum number of retry attempts
        :return: Search results or None
        """
        for attempt in range(max_retries):
            try:
                # Attempt network rotation before each search
                network_rotated = await self.rotate_network_with_admin_check()
                
                # Log the current attempt
                logger.info(f"🔍 Web Search Attempt {attempt + 1}/{max_retries}: {query}")
                
                # Primary search method
                search_results = await self.web_search(query, max_results=300)
                
                # Check if results are valid
                if search_results and len(search_results) > 0:
                    logger.info(f"🌐 Search successful on attempt {attempt + 1}! Found {len(search_results)} results.")
                    return search_results
                
                # If no results, log warning
                logger.warning(f"🚫 Search attempt {attempt + 1} returned no results")
                
                # Exponential backoff with jitter
                await asyncio.sleep(min(2 ** attempt + random.random(), 60))
            
            except Exception as e:
                # Detailed error logging
                logger.error(f"❌ Web search error on attempt {attempt + 1}: {e}")
                
                # Additional error context
                if 'proxy' in str(e).lower():
                    logger.warning("🌐 Potential proxy issue detected. Attempting network rotation.")
                    await self.rotate_network_with_admin_check()
                
                # Exponential backoff with jitter
                await asyncio.sleep(min(2 ** attempt + random.random(), 60))
        
        # Final fallback if all attempts fail
        logger.error(f"🌍 Web search failed after all {max_retries} retry attempts 😞")
        
        # Fallback search methods
        try:
            # Try alternative search method
            fallback_results = await self._fallback_web_search(query)
            if fallback_results:
                logger.info("🔄 Fallback search method successful!")
                return fallback_results
        except Exception as fallback_error:
            logger.error(f"❌ Fallback search method failed: {fallback_error}")
        
        # If all methods fail, return a default response
        return [
            {
                "title": "No Search Results",
                "body": "Unable to find relevant information after multiple search attempts.",
                "link": ""
            }
        ]
    
    async def web_search(self, query: str, max_results: int = 300) -> List[Dict[str, str]]:
        """
        Perform a web search using DuckDuckGo or alternative methods with advanced error handling
        
        :param query: Search query
        :param max_results: Maximum number of search results
        :return: List of search results
        """
        # Check if DDGS is available
        if DDGS is None:
            logger.warning("DuckDuckGo search not available. Using alternative search.")
            return await self._fallback_web_search(query, max_results)
        
        # Attempt multiple proxies and search strategies
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Get a fresh proxy for each attempt
                proxy = proxy_manager.get_proxy()
                proxies = {'http://': proxy, 'https://': proxy} if proxy else None
                
                logger.info(f"Web Search Attempt {attempt + 1}: Query='{query}', Proxy={proxy}")
                
                # Use proxy if available, with configurable timeout
                with DDGS(proxy=proxy, timeout=15) as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                
                # Check if results are meaningful
                if not results:
                    logger.warning(f"No results for query: {query}. Retrying...")
                    if proxy:
                        proxy_manager.release_proxy(proxy, is_bad=True)
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                # Intelligently filter and reduce results
                filtered_results = self._filter_search_results(results)
                
                return filtered_results
            
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"DuckDuckGo search error (Attempt {attempt + 1}): {e}")
                
                # Release problematic proxy
                if proxy:
                    proxy_manager.release_proxy(proxy, is_bad=True)
                
                # Specific error handling
                if "ratelimit" in error_msg or "429" in error_msg:
                    logger.warning("Rate limit detected. Implementing advanced backoff.")
                    await asyncio.sleep(5 * (2 ** attempt))  # More aggressive backoff
                elif "connection" in error_msg or "timeout" in error_msg:
                    logger.warning("Connection issue. Switching proxy.")
                    await asyncio.sleep(2 ** attempt)
                else:
                    break
        
        # If all attempts fail, use fallback search
        logger.warning("All DuckDuckGo search attempts failed. Using fallback method.")
        return await self._fallback_web_search(query, max_results)
    
    async def _fallback_web_search(self, query: str, max_results: int = 300) -> List[Dict[str, str]]:
        """
        Alternative web search method using requests and BeautifulSoup
        
        :param query: Search query
        :param max_results: Maximum number of search results
        :return: List of search results
        """
        try:
            # Encode query for URL
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://www.google.com/search?q={encoded_query}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            # Use proxy if available
            proxy = proxy_manager.get_proxy()
            proxies = {'http://': proxy, 'https://': proxy} if proxy else None
            
            response = requests.get(
                search_url, 
                headers=headers, 
                proxies=proxies, 
                timeout=15
            )
            
            # Check response status
            response.raise_for_status()
            
            # Check if BeautifulSoup is available
            if BeautifulSoup is None:
                logger.warning("BeautifulSoup not available. Returning raw search URL.")
                return [{
                    'title': f'Search results for {query}',
                    'body': f'Search URL: {search_url}',
                    'href': search_url
                }]
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            fallback_results = []
            for result in soup.find_all('div', class_='g')[:max_results]:
                title_elem = result.find('h3')
                snippet_elem = result.find('div', class_='VwiC3b')
                link_elem = result.find('a')
                
                if title_elem and snippet_elem and link_elem:
                    fallback_results.append({
                        'title': title_elem.get_text(strip=True),
                        'body': snippet_elem.get_text(strip=True),
                        'href': link_elem['href']
                    })
            
            return self._filter_search_results(fallback_results)
        
        except requests.exceptions.RequestException as req_error:
            logger.error(f"Fallback search request error: {req_error}")
            
            # Release problematic proxy
            if proxy:
                proxy_manager.release_proxy(proxy, is_bad=True)
            
            return [{
                'title': f'Search error for {query}',
                'body': str(req_error),
                'href': f'https://www.google.com/search?q={query}'
            }]
        
        except Exception as fallback_error:
            logger.error(f"Fallback search unexpected error: {fallback_error}")
            return [{
                'title': f'Search error for {query}',
                'body': str(fallback_error),
                'href': f'https://www.google.com/search?q={query}'
            }]
    
    def _generate_user_mention(self, username: str) -> str:
        """
        Generate a personalized user mention with emojis based on bot's personality
        
        :param username: User's name
        :return: Formatted user mention
        """
        # Personality-based emoji selection
        personality_emojis = {
            'intellectual': ['🧠', '📚', '🤔'],
            'playful': ['🦊', '🌈', '🎉'],
            'artistic': ['🎨', '✨', '🌟'],
            'technical': ['💻', '🤖', '🚀']
        }
        
        # Select emojis based on bot's personality traits
        selected_emojis = []
        if self.personality_traits['communication_style'].get('creativity_level', 0) > 0.8:
            selected_emojis.extend(personality_emojis['artistic'])
        if self.personality_traits['skills'].get('technical', []):
            selected_emojis.extend(personality_emojis['technical'])
        
        # Randomize emoji selection
        emoji = random.choice(selected_emojis) if selected_emojis else '👋'
        
        return f"{emoji} {username}"
    
    def _filter_search_results(self, results: List[Dict[str, str]], max_context_tokens: int = 1000) -> List[Dict[str, str]]:
        """
        Intelligently filter and reduce search results to fit context limits
        
        :param results: Raw search results
        :param max_context_tokens: Maximum tokens for context
        :return: Filtered and reduced search results
        """
        # Sort results by relevance (you can implement more sophisticated sorting)
        sorted_results = sorted(
            results, 
            key=lambda x: len(x.get('body', '')), 
            reverse=True
        )
        
        filtered_results = []
        total_tokens = 0
        
        for result in sorted_results:
            # Estimate token count (rough approximation)
            result_tokens = len(result.get('title', '').split()) + len(result.get('body', '').split())
            
            # Stop if we're approaching token limit
            if total_tokens + result_tokens > max_context_tokens:
                break
            
            filtered_results.append(result)
            total_tokens += result_tokens
        
        return filtered_results
    
    async def process_message(self, message: str, user_id: int, username: str, user_language: str = 'Turkish') -> str:
        """Groq API ile mesaj işleme ve kullanıcı geçmişini dahil etme"""
        try:
            # Perform automatic web search for context enrichment
            search_results = await self.perform_web_search(message, max_retries=3)
            
            # Prepare context messages with web search results
            context_messages = [
                {"role": "system", "content": f"You are {self.current_personality['name']}, an AI assistant with a deep respect for individual perspectives. Respond thoughtfully in {user_language}. Your goal is to understand and support each unique user's journey, free from personal biases."}
            ]
            
            # Add user-specific memory prompt
            user_memory_prompt = self.user_memory_manager.generate_user_memory_prompt(user_id)
            context_messages.append({
                "role": "system", 
                "content": f"Kullanıcının Geçmiş Hafızası:\n{user_memory_prompt}"
            })
            
            # Add web search results to context if available
            if search_results:
                # Limit search context to most relevant results
                search_context = "Recent web search context:\n" + "\n".join([
                    f"- {result['title']}: {result['body'][:200]}..." 
                    for result in search_results[:5]  # Limit to top 5 results
                ])
                context_messages.append({"role": "system", "content": search_context})
            
            # Mevcut mesajı ekle
            context_messages.append({"role": "user", "content": message})
            
            # Groq API çağrısı
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=context_messages,
                max_tokens=int(os.getenv('GROQ_MAX_TOKENS', 4096)),
                temperature=float(os.getenv('GROQ_TEMPERATURE', 0.8))
            )
            
            ai_response = response.choices[0].message.content
            
            # Generate personalized user mention
            user_mention = self._generate_user_mention(username)
            
            # Combine mention and response ONLY ONCE
            ai_response_with_mention = f"{user_mention}, {ai_response}"
            
            # Remove additional username mentions
            def remove_extra_mentions(text, username):
                # Split the text into words
                words = text.split()
                
                # Track if first mention has been processed
                first_mention_processed = False
                
                # Filter out additional mentions
                filtered_words = []
                for word in words:
                    # Check if the word contains the username
                    if username.lower() in word.lower():
                        if not first_mention_processed:
                            filtered_words.append(word)
                            first_mention_processed = True
                    else:
                        filtered_words.append(word)
                
                # Rejoin the words
                return ' '.join(filtered_words)
            
            # Clean the response
            ai_response_with_emoji = self._add_emoji_magic(
                remove_extra_mentions(ai_response_with_mention, username)
            )
            
            return ai_response_with_emoji
        
        except Exception as e:
            logger.error(f"Groq API çağrısında hata: {e}")
            return random.choice(self.responses)

    async def on_ready(self):
        """
        Bot hazır olduğunda çalışan metod
        Sunucudaki tüm kanallara bot ismini sadece bir kez gönderir
        """
        print(f"{self.user} olarak giriş yapıldı!")
        
        # Tüm sunucuları ve kanalları dolaş
        for guild in self.guilds:
            for channel in guild.text_channels:
                try:
                    # Kanala sadece bir kez bot ismini gönder
                    await channel.send(f"Merhaba! Ben {self.personality_traits['name']} 🦊 Bugün size nasıl yardımcı olabilirim?")
                    break  # Her sunucuda sadece bir kanala gönder
                except:
                    continue

    async def on_message(self, message):
        # Kendi mesajlarını ve bot mesajlarını yoksay
        if message.author == self.user or message.author.bot:
            return
        
        # Sadece bot mention edildiğinde yanıt ver
        if self.user.mentioned_in(message):
            try:
                # Botun mentionunu temizle
                user_message = message.content.replace(f'<@{self.user.id}>', '').strip()
                
                # Mesajı işle
                response = await self.process_message(user_message, message.author.id, message.author.name)
                
                # Kullanıcı mesajını ve yanıtı kaydet
                self.user_memory_manager.save_user_conversation(
                    message.author.id, 
                    user_message, 
                    response
                )
                
                # Yanıtı gönder
                await message.channel.send(response)
            
            except Exception as e:
                logger.error(f"Mesaj işleme hatası: {e}")
                await message.channel.send("Bir hata oluştu. Daha sonra tekrar deneyin.")

        await self.process_commands(message)

    def _add_emoji_magic(self, text: str) -> str:
        """
        Restore emoji magic method with a simple implementation
        
        :param text: Input text
        :return: Text with added emoji magic
        """
        emojis = ['✨', '🌟', '🔮', '💫', '🌈']
        import random
        return f"{random.choice(emojis)} {text} {random.choice(emojis)}"

async def validate_proxy_pool():
    """
    Placeholder for proxy pool validation
    Removed detailed validation to simplify startup
    """
    logger.info("Skipping proxy pool validation")
    return True

def main():
    # Validate proxy pool before bot startup
    asyncio.run(validate_proxy_pool())
    
    # Discord intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    # Bot oluştur
    bot = UltraNovativBot(intents=intents)

    # Bot'u çalıştır
    bot.run(os.getenv('DISCORD_BOT_TOKEN'))

if __name__ == "__main__":
    def is_admin():
        """
        Check if the script is running with administrative privileges
        
        :return: Boolean indicating admin status
        """
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False

    main()
