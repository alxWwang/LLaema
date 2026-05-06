import numpy as np
from typing import List, Dict, Optional, Tuple
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import numpy as np
from typing import Dict

from context.topic import TopicManager
from context.session import SessionMemory

# Use local Ollama instance for embeddings (runs on localhost:11434 by default)
MODEL_NAME = "mxbai-embed-large"  # or "mistral" or other Ollama models
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
        
        self.session_memory = SessionMemory(self)
        self.topic_manager = TopicManager(self)
        self.current_topic = "general"

        # Register this instance as the singleton for tool access
        JarvisContextManager._instance = self

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

    @staticmethod
    def mock_data():
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
                        
                        
            # --- New topics with intentional adjacency to existing ones ---
            {
                'topic': 'Astronomy and Astrophysics',
                'content': """Modern astronomy combines observational data from telescopes like Hubble, James Webb, and ground-based arrays with theoretical models of cosmology. Researchers study celestial bodies ranging from exoplanets to distant galaxies, dark matter distributions, and gravitational wave events. Spectroscopy reveals chemical compositions of stars, while redshift measurements help map the expansion of the universe. Black holes and neutron stars test the limits of general relativity, providing extreme laboratories for physics.""",
                'utterances': [
                    'Spectroscopy reveals the chemical composition of distant stars and nebulae.',
                    'Gravitational wave detectors like LIGO measure ripples in spacetime from black hole mergers.',
                    'The James Webb Space Telescope captures infrared light from the earliest galaxies.',
                    'Exoplanet discovery has accelerated through transit photometry and radial velocity measurements of nearby star systems.'
                ]
            },
            {
                'topic': 'Home Cooking',
                'content': """Home cooking encompasses a wide range of culinary traditions, from braising and roasting to baking and fermenting. Mastering basic techniques like making a roux, deglazing a pan, or properly resting meat improves results dramatically. Understanding ingredient ratios in baking is essential for consistent results, while seasoning timing affects how flavors develop. Modern home cooks experiment with sous vide, fermentation, and global cuisines from Korean kimchi to Italian pasta.""",
                'utterances': [
                    'Properly resting meat after cooking allows juices to redistribute through the muscle fibers.',
                    'Sous vide cooking provides precise temperature control for perfect doneness.',
                    'Fermentation transforms ingredients through controlled microbial activity, producing kimchi, sourdough, and yogurt.',
                    'Mastering knife skills and mise en place dramatically improves efficiency and consistency in home kitchens.'
                ]
            },
            {
                'topic': 'Cybersecurity',
                'content': """Cybersecurity protects systems, networks, and data from digital attacks. Modern threats include ransomware, phishing campaigns, supply chain compromises, and zero-day exploits. Defense involves layered approaches: encryption, multi-factor authentication, network segmentation, and continuous monitoring. Red team exercises simulate attackers to find weaknesses, while blue teams defend in real-time. The rise of cloud computing and IoT devices has expanded the attack surface dramatically, requiring new security paradigms.""",
                'utterances': [
                    'Multi-factor authentication adds critical layers beyond password-based security.',
                    'Zero-day exploits target unknown vulnerabilities before patches become available.',
                    'Network segmentation limits lateral movement when a breach occurs.',
                    'Modern ransomware attacks combine encryption with data exfiltration to pressure victims into paying.'
                ]
            },
            {
                'topic': 'Hiking and Trail Running',
                'content': """Hiking and trail running combine endurance with technical terrain navigation. Proper gear includes moisture-wicking clothing, appropriate footwear with grip soles, and navigation tools like maps or GPS. Understanding trail difficulty ratings, weather patterns, and personal pace helps prevent injuries and exhaustion. Long-distance trails like the Pacific Crest Trail or Appalachian Trail attract thousands of thru-hikers annually. Trail running adds cardiovascular intensity to the experience, requiring lighter packs and faster decision-making.""",
                'utterances': [
                    'Proper footwear with aggressive grip soles prevents slips on wet rocks and roots.',
                    'Thru-hiking the Pacific Crest Trail typically takes five to six months of continuous travel.',
                    'Trail running requires careful foot placement to avoid ankle injuries on uneven terrain.',
                    'Layering moisture-wicking base layers with insulating mids and waterproof shells handles variable mountain weather.'
                ]
            },
            {
                'topic': 'Machine Learning',
                'content': """Machine learning enables computers to learn patterns from data without explicit programming. Deep learning uses neural networks with many layers to handle complex tasks like image recognition, natural language processing, and game playing. Training requires large datasets, careful loss function selection, and gradient-based optimization through algorithms like Adam or SGD. Modern approaches include transformer architectures, reinforcement learning, and self-supervised pretraining. Common challenges include overfitting, distribution shift, and interpretability.""",
                'utterances': [
                    'Transformer architectures revolutionized natural language processing through attention mechanisms.',
                    'Gradient descent optimizes neural network weights by following the slope of the loss function.',
                    'Overfitting occurs when models memorize training data rather than learning generalizable patterns.',
                    'Self-supervised pretraining on unlabeled data has dramatically improved downstream task performance across many domains.'
                ]
            },
            {
                'topic': 'Personal Finance',
                'content': """Personal finance covers budgeting, saving, investing, and debt management. Foundational concepts include emergency funds, tax-advantaged retirement accounts like 401(k)s and IRAs, and the difference between assets and liabilities. Index fund investing has become popular for its low fees and broad diversification. Understanding compound interest, tax brackets, and risk tolerance shapes long-term wealth building. Common mistakes include carrying high-interest credit card debt and failing to invest early enough to benefit from time in the market.""",
                'utterances': [
                    'Index funds provide broad market diversification at very low expense ratios.',
                    'Compound interest dramatically rewards starting to invest early in your career.',
                    'Maxing out tax-advantaged retirement accounts like 401(k)s and IRAs reduces taxable income.',
                    'High-interest credit card debt can compound faster than most investments grow, making it a priority to eliminate.'
                ]
            },
            
        ]

        for i in test_data:
            context_manager.add_to_memory(i['content'], i['topic'])
            context_manager.add_to_utterance_memory(i['utterances'], i['topic'])
        return context_manager