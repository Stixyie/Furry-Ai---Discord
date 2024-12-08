import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import discord
from discord.ext import commands
import nltk
import re
import logging

# Türkçe NLP için gerekli NLTK verilerini indir
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Logger ayarları
logger = logging.getLogger(__name__)

class TurkishTextProcessor:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Metni normalize et"""
        # Küçük harfe çevir
        text = text.lower()
        
        # Noktalama işaretlerini temizle
        text = re.sub(r'[^\w\s]', '', text)
        
        # Gereksiz boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Metni kelimelere ayır"""
        return nltk.word_tokenize(text)
    
    @staticmethod
    def get_turkish_stopwords() -> List[str]:
        """Gelişmiş Türkçe stop words listesi"""
        return [
            # Yaygın Türkçe stop words
            'a', 'an', 've', 'veya', 'ama', 'ancak', 'fakat', 
            'ki', 'de', 'da', 'mi', 'mu', 'mü', 'mı', 
            'değil', 'gibi', 'için', 'kadar', 'çok', 'az',
            'ben', 'sen', 'o', 'biz', 'siz', 'onlar',
            'bir', 'bu', 'şu', 'her', 'bazı', 'tüm',
            
            # Ekstra Türkçe stop words
            'ile', 'olarak', 'olup', 'olduğu', 'olduğunu',
            'var', 'yok', 'şimdi', 'artık', 'bile', 'sadece',
            'tam', 'şöyle', 'böyle', 'öyle', 'nasıl', 'niçin',
            
            # Dilbilgisi kelimeleri
            'ise', 'diye', 'ki', 'eğer', 'sanki', 'meğer',
            
            # Zaman ifadeleri
            'şu', 'bu', 'geçen', 'gelecek', 'olan', 'olacak',
            
            # Pekiştireçler
            'dahi', 'zaten', 'bile', 'hatta', 'üstelik', 'ayrıca'
        ]

    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        """Türkçe stop words'leri çıkar"""
        stop_words = TurkishTextProcessor.get_turkish_stopwords()
        return [token for token in tokens if token not in stop_words]
    
    @staticmethod
    def remove_stopwords_from_text(text: str) -> str:
        """Stop words'leri çıkar"""
        try:
            # Metni küçük harfe çevir
            text = text.lower()
            
            # Stop words listesini al
            stop_words = TurkishTextProcessor.get_turkish_stopwords()
            
            # Metni kelimelere ayır
            words = text.split()
            
            # Stop words'leri çıkar
            filtered_words = [word for word in words if word not in stop_words]
            
            # Filtrelenmiş metni birleştir
            return ' '.join(filtered_words)
        
        except Exception as e:
            logger.error(f"Stop words çıkarma hatası: {e}")
            return text
    
    @staticmethod
    def correct_grammar(text: str) -> str:
        """Basit dilbilgisi düzeltmeleri"""
        # Yaygın yazım hatalarını düzelt
        corrections = {
            'yaptımmm': 'yaptım',
            'biliyorumdur': 'biliyorum',
            'çokk': 'çok',
            'merhabaa': 'merhaba'
        }
        
        for error, correction in corrections.items():
            text = text.replace(error, correction)
        
        return text
    
class LearningSystem:
    def __init__(
        self, 
        data_dir: str = None, 
        max_learning_size: int = 5000
    ):
        # Veri dizini yapılandırması
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'data', 
            'learning_data'
        )
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Text processor
        self.text_processor = TurkishTextProcessor()
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            preprocessor=self.text_processor.normalize_text,
            tokenizer=self.text_processor.tokenize,
            stop_words=None  # Manuel stop words kontrolü
        )
        
        # Maksimum öğrenme boyutu
        self.MAX_LEARNING_SIZE = max_learning_size
        
        # Öğrenme kategorileri
        self.learning_categories = {
            'knowledge': [],
            'conversation': [],
            'intent': []
        }
        
        # Sınıflandırıcı
        self.classifier = None
        self.tfidf_vectorizer = None
        
        # Öğrenme istatistikleri
        self.learning_stats = {
            'total_learned_items': 0,
            'category_distribution': {}
        }
        
        # Verileri yükle
        self._load_learning_data()
    
    def _load_learning_data(self):
        """Öğrenme verilerini yükle"""
        for category, items in self.learning_categories.items():
            category_file = os.path.join(self.data_dir, f'{category}_data.json')
            
            if os.path.exists(category_file):
                with open(category_file, 'r', encoding='utf-8') as f:
                    loaded_items = json.load(f)
                    items.extend(loaded_items)
        
        # İstatistikleri güncelle
        self._update_learning_stats()
    
    def add_learning_item(
        self, 
        text: str, 
        category: str = 'conversation',
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Yeni öğrenme öğesi ekle"""
        try:
            # Metni normalize et
            normalized_text = self.text_processor.normalize_text(text)
            corrected_text = self.text_processor.correct_grammar(normalized_text)
            
            # Öğrenme öğesi oluştur
            learning_item = {
                'original_text': text,
                'normalized_text': normalized_text,
                'corrected_text': corrected_text,
                'category': category,
                'metadata': metadata or {}
            }
            
            # Kategoriye ekle
            if category not in self.learning_categories:
                self.learning_categories[category] = []
            
            self.learning_categories[category].append(learning_item)
            
            # İstatistikleri güncelle
            self.learning_stats['total_learned_items'] += 1
            self.learning_stats['category_distribution'][category] = \
                self.learning_stats['category_distribution'].get(category, 0) + 1
            
            logger.info(f"Öğrenme öğesi başarıyla eklendi: {category}")
        
        except Exception as e:
            logger.error(f"Öğrenme öğesi eklenirken hata: {e}")
    
    def _manage_learning_size(self):
        """Öğrenme boyutunu yönet"""
        total_items = sum(len(category) for category in self.learning_categories.values())
        
        if total_items > self.MAX_LEARNING_SIZE:
            # En eski öğeleri çıkar
            self._remove_oldest_items()
    
    def _remove_oldest_items(self):
        """En eski öğrenme öğelerini çıkar"""
        for category, items in self.learning_categories.items():
            # Zamana göre sırala ve en eski öğeleri çıkar
            sorted_items = sorted(
                items, 
                key=lambda x: datetime.fromisoformat(x['timestamp'])
            )
            
            items[:] = sorted_items[len(sorted_items) // 4:]
    
    def _update_learning_stats(self):
        """Öğrenme istatistiklerini güncelle"""
        # Toplam öğrenilen öğe sayısı
        self.learning_stats['total_learned_items'] = sum(
            len(category) for category in self.learning_categories.values()
        )
        
        # Kategori dağılımı
        self.learning_stats['category_distribution'] = {
            category: len(items) 
            for category, items in self.learning_categories.items()
        }
    
    def semantic_search(
        self, 
        query: str, 
        category: str = 'all', 
        top_k: int = 5, 
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Semantik arama"""
        # Aranacak kategorileri belirle
        search_categories = [category] if category != 'all' else list(self.learning_categories.keys())
        
        results = []
        
        for search_category in search_categories:
            items = self.learning_categories[search_category]
            
            # Metinleri topla
            texts = [item['normalized_text'] for item in items]
            
            # TF-IDF embedding
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Query embedding
            query_embedding = self.vectorizer.transform([query])
            
            # Benzerlik hesaplama
            similarities = cosine_similarity(query_embedding, tfidf_matrix)[0]
            
            # En benzer öğeleri bul
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                similarity_score = similarities[idx]
                
                if similarity_score >= min_score:
                    result = {
                        'text': items[idx]['original_text'],
                        'category': search_category,
                        'metadata': items[idx].get('metadata', {}),
                        'similarity_score': similarity_score
                    }
                    results.append(result)
        
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    
    def train_classifier(self, force_retrain: bool = False):
        """Sınıflandırıcıyı güvenli bir şekilde eğit"""
        try:
            # Eğitim verisi kontrolü
            if not self.learning_categories or len(self.learning_categories) < 2:
                logger.warning("Yeterli eğitim verisi yok. En az 2 kategori gerekli. ")
                return False
            
            # Metinleri normalize et
            normalized_texts = []
            labels = []
            
            for category, items in self.learning_categories.items():
                for item in items:
                    normalized_texts.append(item.get('corrected_text', item.get('normalized_text', '')))
                    labels.append(category)
            
            # TF-IDF Vectorizer'ı eğit
            if force_retrain or self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    lowercase=True,
                    stop_words=self.text_processor.get_turkish_stopwords()
                )
            
            # Vektörleri oluştur
            try:
                X = self.tfidf_vectorizer.fit_transform(normalized_texts)
            except Exception as e:
                logger.error(f"TF-IDF dönüşümü sırasında hata: {e}")
                return False
            
            # MLPClassifier'ı eğit
            try:
                self.classifier = MLPClassifier(
                    hidden_layer_sizes=(50, 25),
                    max_iter=500,
                    random_state=42
                )
                self.classifier.fit(X, labels)
            except Exception as e:
                logger.error(f"Sınıflandırıcı eğitimi sırasında hata: {e}")
                return False
            
            logger.info(f"Sınıflandırıcı başarıyla eğitildi. Toplam {len(labels)} örnek kullanıldı. ")
            return True
        
        except Exception as e:
            logger.error(f"Sınıflandırıcı eğitimi sırasında genel hata: {e} ")
            return False
    
    def predict_category(self, text: str) -> str:
        """Metni kategorize et"""
        try:
            # TF-IDF Vectorizer kontrolü
            if self.tfidf_vectorizer is None:
                logger.warning("TF-IDF Vectorizer henüz eğitilmedi. Varsayılan kategoriye dönülüyor.")
                return 'unknown'
            
            # Metni normalize et
            normalized_text = self.text_processor.normalize_text(text)
            corrected_text = self.text_processor.correct_grammar(normalized_text)
            
            # Sınıflandırıcı kontrolü
            if self.classifier is None:
                logger.warning("Sınıflandırıcı henüz eğitilmedi. Varsayılan kategoriye dönülüyor.")
                return 'unknown'
            
            # TF-IDF dönüşümü
            try:
                X = self.tfidf_vectorizer.transform([corrected_text])
            except Exception as e:
                logger.error(f"TF-IDF dönüşümü sırasında hata: {e}")
                return 'unknown'
            
            # Tahmin
            try:
                prediction = self.classifier.predict(X)[0]
                return prediction
            except Exception as e:
                logger.error(f"Kategori tahmini sırasında hata: {e}")
                return 'unknown'
        
        except Exception as e:
            logger.error(f"Kategorilendirme sırasında genel hata: {e}")
            return 'unknown'
    
    def save_learning_data(self):
        """Öğrenme verilerini kaydet"""
        for category, items in self.learning_categories.items():
            category_file = os.path.join(self.data_dir, f'{category}_data.json')
            
            with open(category_file, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=4)
    
    def clear_learning_data(self, category: str = 'all'):
        """Öğrenme verilerini temizle"""
        if category == 'all':
            for category_name in self.learning_categories:
                self.learning_categories[category_name].clear()
        else:
            self.learning_categories[category].clear()
        
        # İstatistikleri sıfırla
        self._update_learning_stats()

async def setup(bot):
    """Discord bot extension için setup fonksiyonu"""
    bot.learning_system = LearningSystem()
    print("Learning System extension yüklendi! ")
