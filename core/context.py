import numpy as np
from typing import List, Dict, Optional
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use local Ollama instance for embeddings (runs on localhost:11434 by default)
MODEL_NAME = "all-minilm:l6-v2"  # or "mistral" or other Ollama models
local_embeddings = OllamaEmbeddings(model=MODEL_NAME)

class JarvisContextManager:

    _instance = None

    def __init__(self):
        dim = len(local_embeddings.embed_query("init"))
        index = faiss.IndexFlatL2(dim)
        self.vector_store = FAISS(
            embedding_function=local_embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        self.topic_vectors = FAISS(
            embedding_function=local_embeddings,
            index=faiss.IndexFlatL2(dim),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        
        self.session_literal_memory = {}

        self.current_topic = "general"

        # Register this instance as the singleton for tool access
        JarvisContextManager._instance = self
    
    def add_session_memory(self, content: str, id:str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split_text(content)
        docs = [Document(page_content=chunk, metadata={"id": id}) for chunk in chunks]
        print(f"[Session Memory] Added {len(docs)} chunks for ID: {id}")
        self.session_literal_memory[id] = docs
    
    def get_session_memory(self, id: str) -> List[Dict]:
        chunk = self.session_literal_memory.get(id, [])[0].page_content if id in self.session_literal_memory and len(self.session_literal_memory[id]) > 0 else None
        self.session_literal_memory[id] = self.session_literal_memory[id][1:] if chunk else []
        return {
            "length": len(self.session_literal_memory.get(id, [])),
            "chunk": chunk,
        }

    def get_topic_keys(self) -> List[str]:
        return list(set(doc.metadata.get("topic") for doc in self.topic_vectors.docstore._dict.values()))

    def _get_embedding(self, text: str) -> np.ndarray:
        return np.array(local_embeddings.embed_query(text))
    
    def add_to_utterance_memory(self, utterances: list[str], topic: str = "general"):
        docs = [Document(page_content=u, metadata={"topic": topic}) for u in utterances]
        self.topic_vectors.add_documents(docs)
    
    def add_to_memory(self, content: str, topic: str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=64)
        chunks = splitter.split_text(content)
        docs = [Document(page_content=chunk, metadata={"topic": topic}) for chunk in chunks]
        self.vector_store.add_documents(docs)

    def query_memory(self, query: str, topic: str = "general", k: int = 5) -> List[Dict]:
        """Query the vector store by topic and semantic similarity."""
        vector = self._get_embedding(query)
        results = self.vector_store.similarity_search_with_score_by_vector(
            vector, k=k, filter={"topic": topic}
        )
        return [{"content": doc.page_content, "topic": doc.metadata.get("topic"), "score": score} for doc, score in results]

    def identify_topic(self, content) -> Optional[str]:
        vector = self._get_embedding(content)
        results = self.topic_vectors.similarity_search_with_score_by_vector(vector, k=20)

        topic_gravity = {}

        for doc, score in results:
            topic = doc.metadata.get("topic")
            weight = 1 / (score + 1e-5)  # Avoid division by zero
            topic_gravity[topic] = topic_gravity.get(topic, 0) + weight

        if not topic_gravity:
            return "general"
        
        sorted_topics = sorted(topic_gravity.items(), key=lambda x: x[1], reverse=True)
    
        top_topic, top_gravity = sorted_topics[0]
        
        if len(sorted_topics) > 1:
            competitors = [g for _, g in sorted_topics[1:]]
            avg_competitor = np.mean(competitors)
            std_competitor = np.std(competitors)
            
            second_gravity = sorted_topics[1][1]
            margin = top_gravity - second_gravity
            ratio = top_gravity / (second_gravity + 1e-9)

            is_clear_winner = top_gravity > (avg_competitor + (1.5 * std_competitor))
            is_distinct = margin > (second_gravity * 0.5)
            is_ratio_winner = ratio >= 1.02
            
            selected_topic = top_topic if ((is_clear_winner and is_distinct) or is_ratio_winner) else "general"
        else:
            selected_topic = top_topic

        print(f"\n--- Topic Election ---")
        for t, g in sorted(topic_gravity.items(), key=lambda x: x[1], reverse=True):
            print(f"Topic: {t:20} | Cumulative Gravity: {g:.4f}")
        print(f"Selected Topic: {selected_topic}\n")
        return selected_topic

if __name__ == "__main__":
    context_manager = JarvisContextManager()
    test_data = [
        {
            'topic': 'Quantum Computing',
            'content': """Quantum computing represents a fundamental shift from classical binary systems. Instead of bits, it uses qubits that exist in superposition, allowing for exponentially more complex calculations. Entanglement connects these qubits across distances, enabling quantum interference to find solutions that classical supercomputers might take centuries to solve. Major players like IBM, Google, and IonQ are racing to achieve quantum supremacy by reducing decoherence and error rates in cryogenic environments.""",
            'utterances': [
                'Qubits utilize superposition and entanglement for complex processing.', 
                'Cryogenic cooling is essential to maintain quantum state stability.', 
                'Quantum interference helps eliminate incorrect computational paths.', 
                'Quantum decoherence remains a major technical hurdle for scaling hardware systems and achieving reliable error correction in modular superconducting circuits.'
            ]
        },
        {
            'topic': 'Sustainable Architecture',
            'content': """Modern sustainable architecture focuses on regenerative design and reducing the carbon footprint of the built environment. This includes the use of cross-laminated timber, passive solar heating, and greywater recycling systems. The 'Living Building Challenge' sets high standards for energy net-zero structures that generate more power than they consume while filtering their own waste. Biophilic design elements also integrate nature into urban spaces to improve mental health.""",
            'utterances': [
                'Regenerative design aims to create buildings with positive environmental impacts.', 
                'Passive solar heating reduces reliance on active mechanical HVAC systems.', 
                'Greywater recycling helps manage onsite water consumption efficiently.', 
                'The integration of biophilic design elements in urban planning significantly reduces heat island effects while fostering psychological well-being through natural connectivity.'
            ]
        },
        {
            'topic': 'Specialty Coffee Craft',
            'content': """The third-wave coffee movement treats coffee as an artisanal product rather than a commodity. This involves transparent sourcing from specific micro-lots, precise roasting profiles to highlight acidity, and meticulous brewing methods like V60 or Chemex. Understanding the terroir—elevation, soil composition, and climate—is crucial for identifying flavor notes ranging from jasmine and bergamot to dark chocolate. Professionals frequently measure extraction yields and TDS.""",
            'utterances': [
                'Terroir influences the specific acidity and flavor profile of the bean.', 
                'Precise extraction measurements ensure consistency in artisanal brewing.', 
                'Micro-lot sourcing allows for complete transparency in the supply chain.', 
                'The application of precise water temperature and Total Dissolved Solids measurement is critical for achieving the ideal balance of chemical flavor extraction in light roasts.'
            ]
        },
        {
            'topic': 'Deep Sea Marine Biology',
            'content': """The hadal zone, located in deep-ocean trenches, remains one of the least explored habitats on Earth. Creatures living here have evolved extreme adaptations to survive immense pressure, perpetual darkness, and near-freezing temperatures. Many organisms utilize bioluminescence to hunt or find mates, while others rely on chemosynthesis near hydrothermal vents where sunlight never reaches. Studying these extremophiles provides insights into the potential for life on moons like Europa.""",
            'utterances': [
                'Bioluminescence is a key adaptation for communication in the deep sea.', 
                'Hydrothermal vents support life through chemical energy instead of sunlight.', 
                'Extreme pressure adaptations are necessary for survival in the hadal zone.', 
                'The physiological mechanisms that prevent cellular collapse under multi-ton pressure levels are currently being researched for breakthroughs in biomedical high-pressure applications.'
            ]
        },
        {
            'topic': 'Amateur Photography',
            'content': """Capturing the perfect shot involves mastering the 'Exposure Triangle': aperture, shutter speed, and ISO. Modern mirrorless cameras offer features like eye-tracking autofocus and in-body image stabilization, making it easier for hobbyists to get sharp images in low light. Compositional techniques such as the rule of thirds or leading lines help guide the viewer's eye through the frame. Post-processing in software like Lightroom allows for recovering details from RAW files.""",
            'utterances': [
                'The Exposure Triangle balances light and motion blur in a photograph.', 
                'Eye-tracking autofocus significantly improves the hit rate for portrait shots.', 
                'RAW files preserve much more dynamic range than compressed JPEG images.', 
                'Utilizing in-body image stabilization allows photographers to capture sharp handheld images at slow shutter speeds previously impossible to achieve without a heavy tripod.'
            ]
        },
        {
            'topic': 'Violin',
            'content': """Four stringed instrument. The violin is a cornerstone of Western classical music, with a rich repertoire spanning from Baroque to contemporary compositions. Its expressive range and technical capabilities have made it a favorite among composers and performers alike. The instrument's design, including its hollow wooden body and f-shaped sound holes, contributes to its distinctive timbre. Mastery of the violin requires years of dedicated practice to develop the necessary bowing techniques and finger dexterity.""",
            'utterances': [
                'Vivaldi’s Four Seasons is a quintessential example of Baroque violin music, showcasing virtuosic techniques and expressive melodies.', 
                'Mozart’s Violin Concerto No. 5 in A major, K. 219, also known as the "Turkish" concerto, is renowned for its lively rhythms and intricate passages.', 
                'Bach’s Partita No. 2 in D minor, BWV 1004, features the famous Chaconne, a monumental work for solo violin.', 
                'The techniques used in Baroque violin music, such as ornamentation and basso continuo, require a deep understanding of historical performance practices.'
            ]
        },
        
    ]

    for i in test_data:
        context_manager.add_to_memory(i['content'], i['topic'])
        context_manager.add_to_utterance_memory(i['utterances'], i['topic'])
    
    while True:
        print("\n--- Jarvis Context Manager ---")
        user_input = input("Enter a query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        topic = context_manager.identify_topic(user_input)
        print(f"Identified topic: {topic}")
        print("Memory results:")
        results = context_manager.query_memory(user_input, topic=topic)
        for idx, res in enumerate(results):
            print(f"{idx+1}. [{res['topic']}] {res['content']} (Score: {res['score']:.4f})")
