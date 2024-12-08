import re
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LanguageProcessor:
    def __init__(
        self, 
        data_dir: str = None, 
        max_context_size: int = 1000
    ):
        # Veri dizini yapılandırması
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'data', 
            'language_data'
        )
        os.makedirs(self.data_dir, exist_ok=True)
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Maksimum içerik boyutu
        self.MAX_CONTEXT_SIZE = max_context_size
        
        # Dil bağlamları
        self.language_contexts = {
            'general': [],
            'specific': [],
            'domain': []
        }
        
        # Dil istatistikleri
        self.language_stats = {
            'total_contexts': 0,
            'context_distribution': {}
        }
        
        # Verileri yükle
        self._load_language_data()
    
    def _load_language_data(self):
        """Dil verilerini yükle"""
        for context_type, contexts in self.language_contexts.items():
            context_file = os.path.join(self.data_dir, f'{context_type}_contexts.json')
            
            if os.path.exists(context_file):
                with open(context_file, 'r', encoding='utf-8') as f:
                    loaded_contexts = json.load(f)
                    contexts.extend(loaded_contexts)
        
        # İstatistikleri güncelle
        self._update_language_stats()
    
    def add_context(
        self, 
        text: str, 
        context_type: str = 'general',
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Yeni dil bağlamı ekle"""
        context_entry = {
            'text': text,
            'context_type': context_type,
            'metadata': metadata or {}
        }
        
        # Bağlama ekle
        self.language_contexts[context_type].append(context_entry)
        
        # Boyut kontrolü
        self._manage_context_size()
        
        # İstatistikleri güncelle
        self._update_language_stats()
    
    def _manage_context_size(self):
        """Bağlam boyutunu yönet"""
        total_contexts = sum(len(contexts) for contexts in self.language_contexts.values())
        
        if total_contexts > self.MAX_CONTEXT_SIZE:
            # En az önemli bağlamları çıkar
            self._remove_least_important_contexts()
    
    def _remove_least_important_contexts(self):
        """En az önemli bağlamları çıkar"""
        for context_type, contexts in self.language_contexts.items():
            # Boyuta göre sırala ve en az önemli bağlamları çıkar
            sorted_contexts = sorted(
                contexts, 
                key=lambda x: len(x['text'])
            )
            
            contexts[:] = sorted_contexts[len(sorted_contexts) // 4:]
    
    def _update_language_stats(self):
        """Dil istatistiklerini güncelle"""
        # Toplam bağlam sayısı
        self.language_stats['total_contexts'] = sum(
            len(contexts) for contexts in self.language_contexts.values()
        )
        
        # Bağlam dağılımı
        self.language_stats['context_distribution'] = {
            context_type: len(contexts) 
            for context_type, contexts in self.language_contexts.items()
        }
    
    def semantic_search(
        self, 
        query: str, 
        context_type: str = 'all', 
        top_k: int = 5, 
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Semantik arama"""
        # Aranacak bağlam türlerini belirle
        search_context_types = [context_type] if context_type != 'all' else list(self.language_contexts.keys())
        
        results = []
        
        for search_context_type in search_context_types:
            contexts = self.language_contexts[search_context_type]
            
            # Metinleri topla
            texts = [context['text'] for context in contexts]
            
            # TF-IDF embedding
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Query embedding
            query_embedding = self.vectorizer.transform([query])
            
            # Benzerlik hesaplama
            similarities = cosine_similarity(query_embedding, tfidf_matrix)[0]
            
            # En benzer bağlamları bul
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                similarity_score = similarities[idx]
                
                if similarity_score >= min_score:
                    result = {
                        'text': contexts[idx]['text'],
                        'context_type': search_context_type,
                        'metadata': contexts[idx].get('metadata', {}),
                        'similarity_score': similarity_score
                    }
                    results.append(result)
        
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    
    def preprocess_text(self, text: str) -> str:
        """Metni ön işleme tabi tut"""
        # Küçük harfe çevir
        text = text.lower()
        
        # Noktalama işaretlerini ve fazladan boşlukları temizle
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save_language_data(self):
        """Dil verilerini kaydet"""
        for context_type, contexts in self.language_contexts.items():
            context_file = os.path.join(self.data_dir, f'{context_type}_contexts.json')
            
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(contexts, f, ensure_ascii=False, indent=4)
    
    def clear_language_data(self, context_type: str = 'all'):
        """Dil verilerini temizle"""
        if context_type == 'all':
            for context_name in self.language_contexts:
                self.language_contexts[context_name].clear()
        else:
            self.language_contexts[context_type].clear()
        
        # İstatistikleri sıfırla
        self._update_language_stats()
