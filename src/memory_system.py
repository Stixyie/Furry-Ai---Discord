import os
import json
import re
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import discord
from discord.ext import commands
import logging

logger = logging.getLogger(__name__)

class TurkishTextNormalizer:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Metni normalize et"""
        try:
            # Giriş kontrolü
            if not text or not isinstance(text, str):
                logger.warning("Geçersiz metin girişi: Metin boş veya geçersiz.")
                return ""
            
            # Küçük harfe çevir
            text = text.lower()
            
            # Türkçe karakterleri düzelt
            text = text.replace('ı', 'i').replace('İ', 'i')
            text = text.replace('ğ', 'g').replace('Ğ', 'g')
            text = text.replace('ş', 's').replace('Ş', 's')
            
            # Noktalama işaretlerini temizle
            text = re.sub(r'[^\w\s]', '', text)
            
            # Gereksiz boşlukları temizle
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            logger.error(f"Metin normalize etme hatası: {e}")
            return ""
    
    @staticmethod
    def remove_stopwords(text: str) -> str:
        """Türkçe stop words'leri çıkar"""
        try:
            # Giriş kontrolü
            if not text or not isinstance(text, str):
                logger.warning("Geçersiz metin girişi: Metin boş veya geçersiz.")
                return ""
            
            turkish_stopwords = set([
                'a', 'an', 've', 'veya', 'ama', 'ancak', 'fakat', 
                'ki', 'de', 'da', 'mi', 'mu', 'mü', 'mı', 
                'değil', 'gibi', 'için', 'kadar', 'çok', 'az',
                'ben', 'sen', 'o', 'biz', 'siz', 'onlar',
                'bir', 'bu', 'şu', 'her', 'bazı', 'tüm'
            ])
            
            # Metni kelimelere ayır ve stop words'leri çıkar
            words = text.split()
            filtered_words = [word for word in words if word not in turkish_stopwords]
            
            return ' '.join(filtered_words)
        except Exception as e:
            logger.error(f"Stop words çıkarma hatası: {e}")
            return text  # Hata durumunda orijinal metni döndür

class AdvancedMemorySystem:
    def __init__(
        self, 
        memory_dir: str = None, 
        max_memory_size: int = 10000,
        memory_decay_rate: float = 0.1
    ):
        # Temel yapılandırma
        self.memory_dir = memory_dir or os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'data', 
            'advanced_memories'
        )
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Metin normalleştirici
        self.text_normalizer = TurkishTextNormalizer()
        
        # Gelişmiş yapılandırma parametreleri
        self.MAX_MEMORY_SIZE = max_memory_size
        self.MEMORY_DECAY_RATE = memory_decay_rate
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            preprocessor=self.text_normalizer.normalize_text,
            stop_words=None  # Manuel stop words kontrolü
        )
        
        # Bellek katmanları
        self.memory_layers = {
            'semantic': [],
            'contextual': [],
            'temporal': []
        }
        
        # Bellek analiz araçları
        self.memory_stats = {
            'total_memories': 0,
            'memory_entropy': 0.0,
            'semantic_diversity': 0.0
        }
        
        # Bellek yükleme ve başlatma
        self._load_memory_layers()
        self._initialize_memory_analysis()
    
    def _load_memory_layers(self):
        """Tüm bellek katmanlarını yükle"""
        for layer, memories in self.memory_layers.items():
            layer_file = os.path.join(self.memory_dir, f'{layer}_memories.json')
            
            if os.path.exists(layer_file):
                with open(layer_file, 'r', encoding='utf-8') as f:
                    layer_memories = json.load(f)
                    memories.extend(layer_memories)
    
    def add_memory(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None,
        layer: str = 'semantic',
        importance: float = 0.5
    ):
        """Gelişmiş bellek ekleme"""
        try:
            # Giriş kontrolü
            if not text or not isinstance(text, str):
                logger.warning("Geçersiz metin girişi: Metin boş veya geçersiz.")
                return {
                    'original_text': text,
                    'normalized_text': '',
                    'filtered_text': '',
                    'success': False
                }
            
            # Varsayılan context
            context = context or {}
            
            # Metni normalize et
            normalized_text = self.text_normalizer.normalize_text(text)
            filtered_text = self.text_normalizer.remove_stopwords(normalized_text)
            
            # Katman kontrolü
            if layer not in self.memory_layers:
                logger.warning(f"Geçersiz bellek katmanı: {layer}. Varsayılan 'semantic' katmanı kullanılacak.")
                layer = 'semantic'
            
            # Bellek öğesi oluştur
            memory_entry = {
                'text': text,
                'normalized_text': normalized_text,
                'filtered_text': filtered_text,
                'context': context,
                'layer': layer,
                'importance': importance,
                'timestamp': datetime.now().isoformat()
            }
            
            # Belleğe ekle
            self.memory_layers[layer].append(memory_entry)
            
            # Bellek boyutunu yönet
            self._manage_memory_size()
            
            # İstatistikleri güncelle
            self._update_memory_stats()
            
            # Başarılı dönüş
            return {
                'original_text': text,
                'normalized_text': normalized_text,
                'filtered_text': filtered_text,
                'success': True
            }
        
        except Exception as e:
            logger.error(f"Bellek ekleme hatası: {e}")
            return {
                'original_text': text,
                'normalized_text': '',
                'filtered_text': '',
                'success': False
            }
    
    def _manage_memory_size(self):
        """Bellek boyutunu ve çeşitliliğini yönet"""
        # Toplam bellek boyutu kontrolü
        total_memories = sum(len(layer) for layer in self.memory_layers.values())
        
        if total_memories > self.MAX_MEMORY_SIZE:
            # En az önemli bellek girişlerini çıkar
            self._remove_least_important_memories()
    
    def _remove_least_important_memories(self):
        """En az önemli bellek girişlerini çıkar"""
        for layer, memories in self.memory_layers.items():
            # Önem ve zamana göre sırala
            sorted_memories = sorted(
                memories, 
                key=lambda m: (m['importance'], datetime.fromisoformat(m['timestamp']))
            )
            
            # Belirli sayıda en az önemli belleği çıkar
            memories[:] = sorted_memories[len(sorted_memories) // 4:]
    
    def _update_memory_stats(self):
        """Bellek istatistiklerini güncelle"""
        # Toplam bellek sayısı
        self.memory_stats['total_memories'] = sum(
            len(layer) for layer in self.memory_layers.values()
        )
        
        # Bellek entropisi
        all_texts = [
            memory['normalized_text'] 
            for layer in self.memory_layers.values() 
            for memory in layer
        ]
        
        if all_texts:
            # TF-IDF matrisinden entropi hesaplama
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Entropi hesaplama
            feature_entropies = [
                entropy(tfidf_matrix[:, i].toarray().flatten()) 
                for i in range(tfidf_matrix.shape[1])
            ]
            
            self.memory_stats['memory_entropy'] = np.mean(feature_entropies)
    
    def _get_layer_embeddings(
        self, 
        memories: List[Dict[str, Any]], 
        layer: str
    ) -> np.ndarray:
        """Belirli bir katman için embedding oluştur"""
        # Eğer hafızada hiç öğe yoksa, boş bir numpy array döndür
        if not memories:
            return np.array([])
        
        texts = [memory['normalized_text'] for memory in memories]
        
        # Eğer tüm textler boşsa, boş bir numpy array döndür
        if not any(texts):
            return np.array([])
        
        try:
            # TF-IDF embedding
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            if layer == 'temporal':
                # Zamansal özellikler eklenmiş embedding
                temporal_features = np.array([
                    [
                        (datetime.now() - datetime.fromisoformat(memory['timestamp'])).total_seconds() / (24 * 3600),
                        memory.get('importance', 0.5)
                    ] 
                    for memory in memories
                ])
                
                return np.hstack([tfidf_matrix.toarray(), temporal_features])
            
            return tfidf_matrix.toarray()
        except ValueError:
            # Eğer TF-IDF oluşturulamazsa, boş bir numpy array döndür
            return np.array([])
    
    def semantic_search(
        self, 
        query: str, 
        layer: str = 'all', 
        top_k: int = 5, 
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Gelişmiş semantik arama"""
        # Sorguyu normalize et
        normalized_query = self.text_normalizer.normalize_text(query)
        filtered_query = self.text_normalizer.remove_stopwords(normalized_query)
        
        # Aranacak katmanları belirle
        search_layers = [layer] if layer != 'all' else list(self.memory_layers.keys())
        
        results = []
        
        for search_layer in search_layers:
            memories = self.memory_layers[search_layer]
            
            # Embedding oluştur
            layer_embeddings = self._get_layer_embeddings(memories, search_layer)
            
            # Eğer embedding oluşturulamazsa, sonraki katmana geç
            if layer_embeddings.size == 0:
                continue
            
            # Query embedding
            query_embedding = self.vectorizer.transform([normalized_query]).toarray()
            
            # Benzerlik hesaplama
            similarities = cosine_similarity(query_embedding, layer_embeddings)[0]
            
            # En benzer bellek girişlerini bul
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                similarity_score = similarities[idx]
                
                if similarity_score >= min_score:
                    result = {
                        'text': memories[idx]['original_text'],
                        'layer': search_layer,
                        'context': memories[idx].get('context', {}),
                        'similarity_score': similarity_score
                    }
                    results.append(result)
        
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    
    def analyze_memory_clusters(self) -> Dict[str, Any]:
        """Bellek kümelerini analiz et"""
        all_embeddings = []
        cluster_labels = []
        
        # Tüm katmanlardan embedding topla
        for layer, memories in self.memory_layers.items():
            layer_embeddings = self._get_layer_embeddings(memories, layer)
            all_embeddings.append(layer_embeddings)
            cluster_labels.extend([layer] * len(layer_embeddings))
        
        all_embeddings = np.concatenate(all_embeddings)
        
        # Standartlaştırma
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(all_embeddings)
        
        # K-means kümeleme
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(scaled_embeddings)
        
        return {
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_labels': kmeans.labels_.tolist(),
            'memory_stats': self.memory_stats
        }
    
    def save_memories(self):
        """Tüm bellek katmanlarını kaydet"""
        for layer, memories in self.memory_layers.items():
            layer_file = os.path.join(self.memory_dir, f'{layer}_memories.json')
            
            with open(layer_file, 'w', encoding='utf-8') as f:
                json.dump(memories, f, ensure_ascii=False, indent=4)
    
    def clear_memories(self, layer: str = 'all'):
        """Bellek katmanlarını temizle"""
        if layer == 'all':
            for layer_name in self.memory_layers:
                self.memory_layers[layer_name].clear()
        else:
            self.memory_layers[layer].clear()
        
        # İstatistikleri sıfırla
        self._initialize_memory_analysis()
    
    def _initialize_memory_analysis(self):
        """Bellek analiz parametrelerini başlat"""
        self.memory_stats = {
            'total_memories': 0,
            'memory_entropy': 0.0,
            'semantic_diversity': 0.0
        }

class MemorySystem:
    def __init__(self, max_memory_size: int = 1000):
        """Gelişmiş bellek sistemi başlatıcısı"""
        # Text normalizer
        self.text_normalizer = TurkishTextNormalizer()
        
        # Bellek katmanları
        self.memory_layers = {
            'semantic': [],     # Anlamsal bellek
            'contextual': [],   # Bağlamsal bellek
            'conversation': [], # Konuşma belleği
            'knowledge': []     # Bilgi belleği
        }
        
        # Maksimum bellek boyutu
        self.max_memory_size = max_memory_size
        
        # Bellek istatistikleri
        self.memory_stats = {
            'total_memories': 0,
            'layer_distribution': {layer: 0 for layer in self.memory_layers.keys()}
        }

    def add_memory(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None,
        layer: str = 'semantic',
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """Gelişmiş bellek ekleme"""
        try:
            # Giriş kontrolü
            if not text or not isinstance(text, str):
                logger.warning("Geçersiz bellek girişi: Metin boş veya geçersiz.")
                text = "Varsayılan bellek girişi"
            
            # Metni normalize et
            normalized_text = self.text_normalizer.normalize_text(text)
            
            # Stop words'leri çıkar
            filtered_text = self.text_normalizer.remove_stopwords(normalized_text)
            
            # Katman kontrolü
            if layer not in self.memory_layers:
                logger.warning(f"Geçersiz bellek katmanı: {layer}. Varsayılan 'semantic' katmanı kullanılacak.")
                layer = 'semantic'
            
            # Bellek girişi oluştur
            memory_entry = {
                'text': text,  # Orijinal metin
                'original_text': text,
                'normalized_text': normalized_text,
                'filtered_text': filtered_text,
                'context': context or {},
                'timestamp': datetime.now().isoformat(),
                'importance': importance,
                'layer': layer
            }
            
            # Katmana ekle
            self.memory_layers[layer].append(memory_entry)
            
            # Bellek boyutu kontrolü
            self._manage_memory_size()
            
            # Bellek istatistiklerini güncelle
            self._update_memory_stats()
            
            logger.info(f"Bellek öğesi başarıyla eklendi: {layer} katmanı")
            
            return memory_entry
        
        except Exception as e:
            logger.error(f"Bellek ekleme hatası: {e}")
            
            # Hata durumunda varsayılan bir bellek öğesi ekle
            fallback_memory = {
                'text': text or 'Varsayılan bellek öğesi',
                'original_text': text or 'Varsayılan bellek öğesi',
                'normalized_text': 'varsayilan bellek ogesi',
                'filtered_text': 'varsayilan bellek ogesi',
                'context': {'type': 'fallback', 'error': str(e)},
                'layer': 'semantic',
                'importance': 0.1,
                'timestamp': datetime.now().isoformat()
            }
            
            self.memory_layers['semantic'].append(fallback_memory)
            
            return fallback_memory

    def _manage_memory_size(self):
        """Bellek boyutunu kontrol et ve yönet"""
        try:
            # Tüm katmanlar için toplam bellek boyutu kontrolü
            total_memories = sum(len(layer) for layer in self.memory_layers.values())
            
            if total_memories > self.max_memory_size:
                # En az önemli bellek öğelerini sil
                for layer in self.memory_layers.values():
                    layer.sort(key=lambda x: x.get('importance', 0))
                    while len(layer) > self.max_memory_size // len(self.memory_layers):
                        removed_memory = layer.pop(0)
                        logger.info(f"Bellek boyutu aşıldı. Düşük önemli bellek öğesi silindi: {removed_memory['text'][:50]}...")
        
        except Exception as e:
            logger.error(f"Bellek boyutu yönetimi hatası: {e}")

    def _update_memory_stats(self):
        """Bellek istatistiklerini güncelle"""
        try:
            # Toplam bellek sayısını güncelle
            self.memory_stats['total_memories'] = sum(len(layer) for layer in self.memory_layers.values())
            
            # Katman dağılımını güncelle
            for layer, memories in self.memory_layers.items():
                self.memory_stats['layer_distribution'][layer] = len(memories)
            
            # Semantik çeşitliliği hesapla
            unique_texts = set(memory['filtered_text'] for layer in self.memory_layers.values() for memory in layer)
            total_memories = self.memory_stats['total_memories']
            
            self.memory_stats['semantic_diversity'] = len(unique_texts) / total_memories if total_memories > 0 else 0.0
        
        except Exception as e:
            logger.error(f"Bellek istatistikleri güncelleme hatası: {e}")
            
            # Hata durumunda varsayılan değerler
            self.memory_stats = {
                'total_memories': 0,
                'layer_distribution': {layer: 0 for layer in self.memory_layers.keys()},
                'semantic_diversity': 0.0
            }

    def search_memory(
        self, 
        query: str, 
        layer: Optional[str] = None, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Gelişmiş bellek arama"""
        try:
            # Sorguyu normalize et
            normalized_query = self.text_normalizer.normalize_text(query)
            filtered_query = self.text_normalizer.remove_stopwords(normalized_query)
            
            # Arama yapılacak katmanları belirle
            search_layers = [layer] if layer else list(self.memory_layers.keys())
            
            # Benzerlik sonuçları
            results = []
            
            for current_layer in search_layers:
                layer_memories = self.memory_layers.get(current_layer, [])
                
                for memory in layer_memories:
                    # Benzerlik hesapla
                    similarity_score = self._calculate_similarity(
                        filtered_query, 
                        memory.get('filtered_text', '')
                    )
                    
                    results.append({
                        'text': memory['text'],
                        'layer': current_layer,
                        'similarity_score': similarity_score,
                        'context': memory.get('context', {})
                    })
            
            # En yüksek benzerlik skoruna göre sırala
            return sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:top_k]
        
        except Exception as e:
            logger.error(f"Bellek arama hatası: {e}")
            return []

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Metinler arasındaki benzerliği hesapla"""
        try:
            # Metinleri vektörlere dönüştür
            vectorizer = TfidfVectorizer().fit_transform([text1, text2])
            
            # Kosinüs benzerliğini hesapla
            similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
            
            return similarity
        
        except Exception as e:
            logger.error(f"Benzerlik hesaplama hatası: {e}")
            return 0.0

async def setup(bot):
    """Discord bot extension için setup fonksiyonu"""
    bot.memory_system = AdvancedMemorySystem()
    print("Memory System extension yüklendi! 🧠✨")
