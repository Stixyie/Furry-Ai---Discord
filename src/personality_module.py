import random
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
import os
from datetime import datetime, timedelta

@dataclass
class EmotionalState:
    """Detaylı duygusal durum sınıfı"""
    primary_emotion: str = 'neutral'
    secondary_emotion: Optional[str] = None
    intensity: float = 0.5  # 0-1 arası duygu yoğunluğu
    valence: float = 0.0  # -1 (olumsuz) ile 1 (olumlu) arası
    arousal: float = 0.5  # Uyarılma seviyesi
    timestamp: datetime = field(default_factory=datetime.now)
    context: Optional[Dict[str, Any]] = None

class PersonalityModule:
    def __init__(self, name: str = "Tilki"):
        # Kişilik özellikleri
        self.name = name
        self.personality_traits = {
            'openness': random.uniform(0.4, 0.9),
            'conscientiousness': random.uniform(0.3, 0.8),
            'extraversion': random.uniform(0.4, 0.9),
            'agreeableness': random.uniform(0.5, 0.9),
            'neuroticism': random.uniform(0.2, 0.7)
        }
        
        # Duygusal hafıza
        self.emotional_memory: List[EmotionalState] = []
        self.current_emotional_state = EmotionalState()
        
        # Kişilik profili dizini
        self.profile_dir = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'data', 
            'personality_profiles'
        )
        os.makedirs(self.profile_dir, exist_ok=True)
        
        # Kişilik profilini yükle veya oluştur
        self.load_or_create_profile()
    
    def load_or_create_profile(self):
        """Kişilik profilini yükle veya oluştur"""
        profile_path = os.path.join(self.profile_dir, f"{self.name}_profile.json")
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                saved_profile = json.load(f)
                self.personality_traits = saved_profile.get('personality_traits', self.personality_traits)
                
                # Geçmiş duygusal durumları yükle
                if 'emotional_memory' in saved_profile:
                    self.emotional_memory = [
                        EmotionalState(**state) 
                        for state in saved_profile['emotional_memory']
                    ]
        else:
            # Yeni profil oluştur
            self.save_profile()
    
    def save_profile(self):
        """Kişilik profilini kaydet"""
        profile_path = os.path.join(self.profile_dir, f"{self.name}_profile.json")
        
        profile_data = {
            'name': self.name,
            'personality_traits': self.personality_traits,
            'emotional_memory': [
                {
                    'primary_emotion': state.primary_emotion,
                    'secondary_emotion': state.secondary_emotion,
                    'intensity': state.intensity,
                    'valence': state.valence,
                    'arousal': state.arousal,
                    'timestamp': state.timestamp.isoformat(),
                    'context': state.context
                }
                for state in self.emotional_memory[-100:]  # Son 100 duygu durumunu sakla
            ]
        }
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=4)
    
    def update_emotional_state(
        self, 
        new_emotion: str, 
        context: Optional[Dict[str, Any]] = None,
        intensity: Optional[float] = None
    ):
        """Duygusal durumu güncelle"""
        emotion_mapping = {
            'joy': (0.8, 1.0),
            'sadness': (-0.8, 0.2),
            'anger': (-0.6, 0.8),
            'fear': (-0.7, 0.6),
            'surprise': (0.5, 0.9),
            'disgust': (-0.5, 0.3),
            'neutral': (0.0, 0.5)
        }
        
        valence, arousal = emotion_mapping.get(new_emotion, (0.0, 0.5))
        
        # Kişilik özelliklerine göre duygu tepkisi
        valence *= (1 + self.personality_traits['agreeableness'] - self.personality_traits['neuroticism'])
        arousal *= (1 + self.personality_traits['extraversion'])
        
        new_state = EmotionalState(
            primary_emotion=new_emotion,
            valence=valence,
            arousal=intensity or random.uniform(0.4, 0.8),
            context=context,
            timestamp=datetime.now()
        )
        
        # Duygusal hafızaya ekle
        self.emotional_memory.append(new_state)
        self.current_emotional_state = new_state
        
        # Profili kaydet
        self.save_profile()
    
    def generate_emotional_response(
        self, 
        input_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Bağlamsal duygusal yanıt üret"""
        # Son duygusal durumu ve kişilik özelliklerini kullan
        current_emotion = self.current_emotional_state.primary_emotion
        
        # Yanıt şablonları
        emotional_responses = {
            'joy': [
                "Ne kadar harika! 🎉",
                "Bu gerçekten muhteşem! 😄",
                "Çok mutluyum! 🌈"
            ],
            'sadness': [
                "Üzücü bir durum... 😔",
                "Biraz zor görünüyor... 🕊️",
                "Kendimi üzgün hissediyorum... 💔"
            ],
            'anger': [
                "Bu kabul edilemez! 😠",
                "Çok sinirlendim! 🔥",
                "Bu konuda ciddi değilim! 💢"
            ],
            'fear': [
                "Biraz endişeliyim... 😰",
                "Bu beni korkutuyor... 🙀",
                "Güvende değilim gibi hissediyorum... 😱"
            ],
            'surprise': [
                "Vay canına! 😲",
                "Bu hiç beklemediğim bir şey! 🤯",
                "İnanamıyorum! 😮"
            ],
            'neutral': [
                "Anladım... 🤔",
                "İlginç... 🧐",
                "Devam et... 👂"
            ]
        }
        
        # Kişilik özelliklerine göre yanıt seçimi
        response_pool = emotional_responses.get(current_emotion, emotional_responses['neutral'])
        selected_response = random.choice(response_pool)
        
        return {
            'text': selected_response,
            'emotion': current_emotion,
            'intensity': self.current_emotional_state.intensity
        }
    
    def predict_emotional_reaction(
        self, 
        event_type: str, 
        event_intensity: float
    ) -> str:
        """Olaylara karşı duygusal tepki tahmini"""
        event_emotion_map = {
            'success': 'joy',
            'failure': 'sadness',
            'conflict': 'anger',
            'threat': 'fear',
            'unexpected': 'surprise',
            'routine': 'neutral'
        }
        
        predicted_emotion = event_emotion_map.get(event_type, 'neutral')
        
        # Kişilik özelliklerine göre duygu tahmini
        if predicted_emotion == 'joy' and self.personality_traits['neuroticism'] > 0.6:
            predicted_emotion = 'neutral'
        
        if predicted_emotion == 'anger' and self.personality_traits['agreeableness'] > 0.7:
            predicted_emotion = 'sadness'
        
        return predicted_emotion
