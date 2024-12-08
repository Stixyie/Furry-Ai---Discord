import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
from groq import Client as Groq
from dotenv import load_dotenv

load_dotenv()

class GroqMemoryManager:
    def __init__(self, user_id: str = None):
        """
        Initialize memory manager with unlimited, persistent memory storage
        
        Args:
            user_id (str, optional): Unique identifier for the user. Defaults to generating a new UUID.
        """
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.user_id = user_id or str(uuid.uuid4())
        
        # Unlimited memory storage configuration
        self.memories_dir = os.path.join('advanced_memories', self.user_id)
        os.makedirs(self.memories_dir, exist_ok=True)
        
        # Unlimited memory settings
        self.max_memory_fragments = float('inf')  # Literally infinite memory
        self.memory_retention_strategy = 'comprehensive'

    def _generate_memory_filename(self, category: str) -> str:
        """
        Generate a unique filename for memory storage with comprehensive naming
        
        Args:
            category (str): Memory category
        
        Returns:
            str: Full path to memory file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_id = str(uuid.uuid4())
        filename = f"{category}_{timestamp}_{unique_id}.json"
        return os.path.join(self.memories_dir, filename)

    def store_memory(self, memory: Dict[str, Any], category: str = 'comprehensive') -> None:
        """
        Store memory with comprehensive metadata and unlimited storage
        
        Args:
            memory (Dict[str, Any]): Memory content to store
            category (str, optional): Memory category. Defaults to 'comprehensive'.
        """
        # Enrich memory with extensive metadata
        memory_entry = {
            'memory_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'user_id': self.user_id,
            'content': memory,
            'metadata': {
                'source': 'groq_memory_system',
                'version': '1.0',
                'retention_priority': 'high'
            }
        }
        
        filepath = self._generate_memory_filename(category)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory_entry, f, ensure_ascii=False, indent=2)

    def retrieve_memories(self, 
                          category: str = None, 
                          limit: int = None, 
                          start_date: datetime = None, 
                          end_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Advanced memory retrieval with comprehensive filtering
        
        Args:
            category (str, optional): Filter memories by category
            limit (int, optional): Limit number of memories (None means unlimited)
            start_date (datetime, optional): Retrieve memories from this date
            end_date (datetime, optional): Retrieve memories until this date
        
        Returns:
            List[Dict[str, Any]]: List of memory fragments
        """
        memories = []
        
        # Sort files to get most recent first
        memory_files = sorted(
            [f for f in os.listdir(self.memories_dir) if f.endswith('.json')], 
            reverse=True
        )
        
        for filename in memory_files:
            filepath = os.path.join(self.memories_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                
                # Apply filters
                if category and memory.get('category') != category:
                    continue
                
                if start_date:
                    memory_date = datetime.fromisoformat(memory.get('timestamp', ''))
                    if memory_date < start_date:
                        continue
                
                if end_date:
                    memory_date = datetime.fromisoformat(memory.get('timestamp', ''))
                    if memory_date > end_date:
                        continue
                
                memories.append(memory)
                
                # Optional limit (None means unlimited)
                if limit and len(memories) >= limit:
                    break
            
            except json.JSONDecodeError:
                print(f"Error reading memory file: {filename}")
        
        return memories

    def generate_context_prompt(self, recent_memories: List[Dict[str, Any]] = None) -> str:
        """
        Generate an ultra-comprehensive context prompt
        
        Args:
            recent_memories (List[Dict[str, Any]], optional): List of recent memories
        
        Returns:
            str: Extremely detailed context prompt
        """
        if not recent_memories:
            recent_memories = self.retrieve_memories(limit=10)
        
        memory_context = "\n".join([
            f"🔹 Memory [{m.get('timestamp', 'Unknown Time')}]: {json.dumps(m.get('content', 'No details'), ensure_ascii=False)}"
            for m in recent_memories
        ])
        
        return f"""🦊 ULTIMATE FURRY FOX AI MEMORY CONTEXT 🦊

🌈 User Identification:
- User ID: {self.user_id}
- Interaction Timestamp: {datetime.now().isoformat()}

🧠 Personality Core:
- Name: FoxyAI
- Type: Advanced Multilingual Furry Fox AI
- Memory Strategy: Comprehensive Unlimited Retention

📚 Memory Fragments:
{memory_context}

🌟 Interaction Guidelines:
- ABSOLUTE memory retention
- Adapt to user's language and context INSTANTLY
- Use MAXIMUM emoji expression
- Provide DEEP, INTELLIGENT responses
- NEVER forget ANY detail
- Transform every interaction into a memorable experience

🔮 Communication Directive:
Engage with MAXIMUM creativity, intelligence, and playful furry fox energy! 
Every word is a brushstroke in our shared conversational masterpiece! 🐾✨
"""

    def send_to_groq(self, messages: List[Dict[str, str]]) -> str:
        """
        Send messages to Groq API with ultimate memory context
        
        Args:
            messages (List[Dict[str, str]]): Conversation messages
        
        Returns:
            str: API response
        """
        recent_memories = self.retrieve_memories(limit=20)  # More memories for context
        context_prompt = self.generate_context_prompt(recent_memories)
        
        system_message = {
            "role": "system", 
            "content": context_prompt
        }
        
        full_messages = [system_message] + messages
        
        response = self.client.chat.completions.create(
            model=os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile'),
            messages=full_messages,
            max_tokens=4096,  # Maximum token support
            temperature=0.8,  # Creative but controlled
            top_p=0.9  # High-quality diverse responses
        )
        
        # Store the AI's response as a memory
        self.store_memory({
            "content": response.choices[0].message.content,
            "role": "assistant",
            "interaction_metadata": {
                "model": response.model,
                "tokens_used": response.usage.total_tokens
            }
        }, category="conversation")
        
        return response.choices[0].message.content

# Unlimited memory usage example
if __name__ == "__main__":
    memory_manager = GroqMemoryManager()
    
    # Example comprehensive memory storage
    memories_to_store = [
        {"topic": "First conversation", "details": "Initial interaction"},
        {"topic": "Technical discussion", "details": "Advanced AI concepts"},
        {"topic": "Personal story", "details": "User's background"}
    ]
    
    for memory in memories_to_store:
        memory_manager.store_memory(memory)
    
    # Retrieve and print all memories
    all_memories = memory_manager.retrieve_memories()
    print(f"Total Memories Stored: {len(all_memories)}")
    for mem in all_memories:
        print(json.dumps(mem, indent=2, ensure_ascii=False))
