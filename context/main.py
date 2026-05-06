from context.context import JarvisContextManager
import sys
from io import StringIO

def main():
    context_manager = JarvisContextManager.mock_data()
    while True:
        print("\n--- Jarvis Context Manager ---")
        user_input = input("Enter a query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        topic = context_manager.topic_manager.identify_topic(user_input)
        print(f"Identified topic: {topic}")


def run_test(tests: list):
    # Capture all prints to file
    output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = output
    
    cm = JarvisContextManager.mock_data()
    run_tests(cm, tests)
    
    # Restore stdout and save to file
    sys.stdout = original_stdout
    with open("test_results.txt", "w") as f:
        f.write(output.getvalue())
    print(output.getvalue())
    

def run_tests(context_manager, tests: list, verbose=True):
    """Run the test suite and return categorized results."""
    results = []
    for idx, test in enumerate(tests, 1):
        if not test["query"]:
            continue
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"TEST {idx}/{len(tests)} [{test['category']}]")
            print(f"Query: {test['query']!r}")
            print(f"{'='*70}")
        
        try:
            actual = context_manager.topic_manager.identify_topic(test["query"])
        except Exception as e:
            results["fail"].append({**test, "error": str(e)})
            print(f"❌ ERROR: {e}")
            continue
        
        expected = test["expected"]
        actual_topics = set(actual.keys())
        expected_topics = set(expected.keys())

        # ---- Scoring ----
        if not expected_topics:
            # Empty-target case: 1.0 if correctly rejected, 0.0 otherwise
            score = 1.0 if not actual_topics else 0.0
            label = "1.0 ✅ correctly rejected" if score == 1.0 else f"0.0 ❌ should be empty, got {actual_topics}"
        elif not actual_topics:
            # Expected something but got nothing
            score = 0.0
            label = f"0.0 ❌ no topics returned, expected {expected_topics}"
        elif actual_topics == expected_topics:
            # Exact match
            score = 1.0
            label = f"1.0 ✅ exact match"
        elif expected_topics.issubset(actual_topics):
            # All expected present, but extra topics detected
            extra = actual_topics - expected_topics
            point = 1 - 0.3 * len(extra)  # Deduct 0.3 points per extra topic
            score = max(point, 0.0)  # Ensure score doesn't go below 0
            label = f"{score:.1f} ➕ correct + extra {extra}"
        elif actual_topics & expected_topics:
            # Some expected topics present, some missing
            hit = actual_topics & expected_topics
            missing = expected_topics - actual_topics
            point = 1 - 0.5 * len(missing)  # Deduct 0.5 points for any missing topic
            score = max(point, 0.0) 
            label = f"{score:.1f} 🟡 partial — hit {hit}, missed {missing}"
        else:
            # No overlap at all
            score = 0.0
            label = f"0.0 ❌ wrong — expected {expected_topics}, got {actual_topics}"

        results.append({**test, "actual": actual, "score": score})
        print(f"\n>>> {label}")
        print(f">>> Notes: {test['notes']}")

    # ---- Summary ----
    from collections import defaultdict
    total = len(results)
    total_score = sum(r["score"] for r in results)
    max_score = float(total)

    print(f"\n\n{'#'*70}")
    print(f"# RESULTS SUMMARY")
    print(f"{'#'*70}")
    print(f"Score:   {total_score:.1f} / {max_score:.1f}  ({100 * total_score / max_score:.1f}%)")
    counts = {1.0: 0, 0.7: 0, 0.5: 0, 0.0: 0}
    for r in results:
        counts[r["score"]] += 1
    print(f"  1.0 exact  : {counts[1.0]:>3}  ({100*counts[1.0]/total:.0f}%)")
    print(f"  0.7 extra  : {counts[0.7]:>3}  ({100*counts[0.7]/total:.0f}%)")
    print(f"  0.5 partial: {counts[0.5]:>3}  ({100*counts[0.5]/total:.0f}%)")
    print(f"  0.0 wrong  : {counts[0.0]:>3}  ({100*counts[0.0]/total:.0f}%)")

    # Breakdown by category
    by_cat = defaultdict(lambda: {"score": 0.0, "total": 0})
    for r in results:
        by_cat[r["category"]]["score"] += r["score"]
        by_cat[r["category"]]["total"] += 1

    print(f"\nBy category:")
    print(f"  {'Category':<22} {'Score':>7}  {'Max':>5}  {'%':>6}")
    print(f"  {'-'*42}")
    for cat in sorted(by_cat.keys()):
        c = by_cat[cat]
        pct = 100 * c["score"] / c["total"]
        print(f"  {cat:<22} {c['score']:>7.1f}  {c['total']:>5.1f}  {pct:>5.0f}%")

    # Show non-perfect results
    non_perfect = [r for r in results if r["score"] < 1.0]
    if non_perfect:
        print(f"\n\n{'!'*70}")
        print(f"# NON-PERFECT ({len(non_perfect)} tests)")
        print(f"{'!'*70}")
        for r in non_perfect:
            print(f"\n[{r['category']}] score={r['score']}  {r['query']!r}")
            print(f"  Expected: {r['expected']}")
            print(f"  Actual:   {r.get('actual', 'ERROR: ' + r.get('error', ''))}")
            print(f"  Notes:    {r['notes']}")

    return results

if __name__ == "__main__":
    tests = [
        # ============ TIER 1: Single Topic, Strong Signal (Original 6) ============
        {"query": "qubits and superposition", "expected": {"Quantum Computing": 3}, "category": "single_dominant", "notes": "Direct vocabulary match"},
        {"query": "brewing pour over with a chemex", "expected": {"Specialty Coffee Craft": 3}, "category": "single_dominant", "notes": "Coffee jargon"},
        {"query": "shutter speed and aperture settings", "expected": {"Amateur Photography": 3}, "category": "single_dominant", "notes": "Photo terminology"},
        {"query": "hadal zone hydrothermal vents", "expected": {"Deep Sea Marine Biology": 3}, "category": "single_dominant", "notes": "Marine bio terms"},
        {"query": "cross laminated timber and passive solar", "expected": {"Sustainable Architecture": 3}, "category": "single_dominant", "notes": "Architecture terms"},
        {"query": "Vivaldi Four Seasons concerto", "expected": {"Violin": 3}, "category": "single_dominant", "notes": "Classical music ref"},

        # ============ TIER 1B: Single Topic, New Topics ============
        {"query": "James Webb infrared galaxy observations", "expected": {"Astronomy and Astrophysics": 3}, "category": "single_dominant", "notes": "Astronomy direct"},
        {"query": "sous vide steak with proper resting time", "expected": {"Home Cooking": 3}, "category": "single_dominant", "notes": "Cooking technique"},
        {"query": "ransomware attack with data exfiltration", "expected": {"Cybersecurity": 3}, "category": "single_dominant", "notes": "Cybersec direct"},
        {"query": "thru-hiking the Pacific Crest Trail", "expected": {"Hiking and Trail Running": 3}, "category": "single_dominant", "notes": "Hiking direct"},
        {"query": "transformer attention mechanism architecture", "expected": {"Machine Learning": 3}, "category": "single_dominant", "notes": "ML terminology"},
        {"query": "compound interest in retirement accounts", "expected": {"Personal Finance": 3}, "category": "single_dominant", "notes": "Finance direct"},

        # ============ TIER 2: Multi-Topic, Clear Boundaries ============
        {"query": "taking a picture of an octopus playing the violin", "expected": {"Violin": 3, "Amateur Photography": 2, "Deep Sea Marine Biology": 1}, "category": "multi_topic", "notes": "Three distinct concepts"},
        {"query": "photographing the deep sea creatures with bioluminescence", "expected": {"Deep Sea Marine Biology": 3, "Amateur Photography": 2}, "category": "multi_topic", "notes": "Two clear topics"},
        {"query": "designing a sustainable building for quantum research labs", "expected": {"Sustainable Architecture": 3, "Quantum Computing": 2}, "category": "multi_topic", "notes": "Architecture + tech"},
        {"query": "brewing coffee while hiking the Appalachian Trail", "expected": {"Specialty Coffee Craft": 3, "Hiking and Trail Running": 2}, "category": "multi_topic", "notes": "Coffee + outdoor"},
        {"query": "machine learning models for cybersecurity threat detection", "expected": {"Machine Learning": 3, "Cybersecurity": 2}, "category": "multi_topic", "notes": "Two technical fields"},
        {"query": "photographing exoplanets with telescope cameras", "expected": {"Astronomy and Astrophysics": 3, "Amateur Photography": 2}, "category": "multi_topic", "notes": "Astronomy + photo"},
        {"query": "investing my savings into quantum computing startups", "expected": {"Personal Finance": 3, "Quantum Computing": 2}, "category": "multi_topic", "notes": "Finance + tech"},
        {"query": "fermenting kimchi while listening to violin music", "expected": {"Home Cooking": 3, "Violin": 2}, "category": "multi_topic", "notes": "Cooking + music"},

        # ============ TIER 3: Adjacent Topics (Hard Disambiguation) ============
        # New topics deliberately designed to look similar to existing ones
        {"query": "spectroscopy of distant stars", "expected": {"Astronomy and Astrophysics": 3}, "category": "adjacent", "notes": "Could leak to Quantum"},
        {"query": "encryption and quantum computing threats", "expected": {"Cybersecurity": 3, "Quantum Computing": 2}, "category": "adjacent", "notes": "Both technical, related"},
        {"query": "neural network gradient descent optimization", "expected": {"Machine Learning": 3}, "category": "adjacent", "notes": "Could leak to Quantum"},
        {"query": "fermenting sourdough at home", "expected": {"Home Cooking": 3}, "category": "adjacent", "notes": "Could leak to Coffee (brewing/fermenting)"},
        {"query": "trail running with a heart rate monitor", "expected": {"Hiking and Trail Running": 3}, "category": "adjacent", "notes": "Solo topic"},
        {"query": "index fund expense ratios", "expected": {"Personal Finance": 3}, "category": "adjacent", "notes": "Solo topic"},
        {"query": "deep learning for image classification", "expected": {"Machine Learning": 3}, "category": "adjacent", "notes": "Could touch Photography"},

        # ============ TIER 4: Vocabulary Gaps ============
        {"query": "octopus", "expected": {"Deep Sea Marine Biology": 3}, "category": "vocab_gap", "notes": "Single word, not in utterances"},
        {"query": "Stradivarius", "expected": {"Violin": 3}, "category": "vocab_gap", "notes": "Embedding generalization"},
        {"query": "espresso", "expected": {"Specialty Coffee Craft": 3}, "category": "vocab_gap", "notes": "Single word reject"},
        {"query": "the Stradivarius violin maker history", "expected": {"Violin": 3}, "category": "vocab_gap", "notes": "Stradivarius + context"},
        {"query": "an octopus in the hadal zone", "expected": {"Deep Sea Marine Biology": 3}, "category": "vocab_gap", "notes": "Octopus + utterance vocab"},
        {"query": "supernova neutron star collapse", "expected": {"Astronomy and Astrophysics": 3}, "category": "vocab_gap", "notes": "Supernova not in utterances"},
        {"query": "phishing email attachments", "expected": {"Cybersecurity": 3}, "category": "vocab_gap", "notes": "Phishing not in utterances"},
        {"query": "Roth IRA contribution limits", "expected": {"Personal Finance": 3}, "category": "vocab_gap", "notes": "Roth not specifically in utterances"},
        {"query": "Bordeaux red wine pairings", "expected": {}, "category": "vocab_gap", "notes": "Wine has no topic"},
        {"query": "proper knife skills for vegetables", "expected": {"Home Cooking": 3}, "category": "vocab_gap", "notes": "Knife skills + vegetables"},

        # ============ TIER 5: Adversarial Noise ============
        {"query": "coffee computer", "expected": {}, "category": "noise", "notes": "Two random words"},
        {"query": "the the the", "expected": {}, "category": "noise", "notes": "Stopword spam"},
        {"query": "hello", "expected": {}, "category": "noise", "notes": "Greeting"},
        {"query": "what is it", "expected": {}, "category": "noise", "notes": "Pronoun reference"},
        {"query": "yes please", "expected": {}, "category": "noise", "notes": "Affirmation"},
        {"query": "tell me more about that", "expected": {}, "category": "noise", "notes": "Continuation phrase"},
        {"query": "and then what", "expected": {}, "category": "noise", "notes": "Anaphoric reference"},
        {"query": "okay sure", "expected": {}, "category": "noise", "notes": "Acknowledgment"},
        {"query": "I dont know", "expected": {}, "category": "noise", "notes": "Uncertainty"},
        {"query": "remind me later please", "expected": {}, "category": "noise", "notes": "Deferral"},

        # ============ TIER 6: Out-of-Distribution ============
        {"query": "best way to sharpen a chef knife", "expected": {"Home Cooking": 3}, "category": "out_of_dist", "notes": "Now in-domain (Home Cooking)"},
        {"query": "training my golden retriever to sit", "expected": {}, "category": "out_of_dist", "notes": "No animal training topic"},
        {"query": "how do I file my taxes correctly", "expected": {"Personal Finance": 3}, "category": "out_of_dist", "notes": "Now in-domain"},
        {"query": "recipe for chocolate chip cookies", "expected": {"Home Cooking": 3}, "category": "out_of_dist", "notes": "Now in-domain"},
        {"query": "treating skin conditions with topical medication", "expected": {}, "category": "out_of_dist", "notes": "Medical, OOD"},
        {"query": "how to negotiate a salary increase", "expected": {}, "category": "out_of_dist", "notes": "Career, OOD"},
        {"query": "Mandarin Chinese grammar rules", "expected": {}, "category": "out_of_dist", "notes": "Linguistics, OOD"},
        {"query": "writing a novel with multiple POV characters", "expected": {}, "category": "out_of_dist", "notes": "Fiction writing, OOD"},

        # ============ TIER 7: Edge Cases ============
        {"query": "q", "expected": {}, "category": "edge", "notes": "Single character"},
        {"query": "quantum quantum quantum quantum quantum", "expected": {"Quantum Computing": 3}, "category": "edge", "notes": "Repetition"},
        {"query": "I want to learn quantum computing photography hiking machine learning and cooking all at once", "expected": {"Quantum Computing": 3, "Amateur Photography": 2, "Machine Learning": 1}, "category": "edge", "notes": "5+ topics, k=3 ceiling"},
        {"query": "a a a a a a a a a a", "expected": {}, "category": "edge", "notes": "Stopword spam"},
        {"query": "?????", "expected": {}, "category": "edge", "notes": "Punctuation only"},
        {"query": "violin VIOLIN ViOlIn", "expected": {"Violin": 3}, "category": "edge", "notes": "Case variations"},
        {"query": "1234567890", "expected": {}, "category": "edge", "notes": "Numbers only"},

        # ============ TIER 8: Paraphrase Robustness ============
        {"query": "quietly listening to a string instrument concerto", "expected": {"Violin": 3}, "category": "paraphrase", "notes": "Violin paraphrased"},
        {"query": "eco-friendly green construction methods", "expected": {"Sustainable Architecture": 3}, "category": "paraphrase", "notes": "Sustainable synonyms"},
        {"query": "underwater creatures that glow in darkness", "expected": {"Deep Sea Marine Biology": 3}, "category": "paraphrase", "notes": "Bioluminescence paraphrase"},
        {"query": "capturing images with low light cameras", "expected": {"Amateur Photography": 3}, "category": "paraphrase", "notes": "Photography paraphrase"},
        {"query": "stargazing through a backyard telescope", "expected": {"Astronomy and Astrophysics": 3}, "category": "paraphrase", "notes": "Astronomy paraphrase"},
        {"query": "preventing computer break-ins and data theft", "expected": {"Cybersecurity": 3}, "category": "paraphrase", "notes": "Cybersec paraphrase"},
        {"query": "saving money for retirement years", "expected": {"Personal Finance": 3}, "category": "paraphrase", "notes": "Finance paraphrase"},
        {"query": "going for long walks in the mountains", "expected": {"Hiking and Trail Running": 3}, "category": "paraphrase", "notes": "Hiking paraphrase"},
        {"query": "AI systems that learn from examples", "expected": {"Machine Learning": 3}, "category": "paraphrase", "notes": "ML paraphrase"},
        {"query": "preparing meals at home from scratch", "expected": {"Home Cooking": 3}, "category": "paraphrase", "notes": "Cooking paraphrase"},

        # ============ TIER 9: Chunking Stress ============
        {"query": "the violin and the camera and the qubit", "expected": {"Violin": 3, "Amateur Photography": 2, "Quantum Computing": 1}, "category": "chunk_stress", "notes": "Three nouns isolated"},
        {"query": "violin violin violin violin", "expected": {"Violin": 3}, "category": "chunk_stress", "notes": "Repetition with overlap"},
        {"query": "I photographed a building made of timber", "expected": {"Sustainable Architecture": 3, "Amateur Photography": 2}, "category": "chunk_stress", "notes": "Two topics, one keyword each"},
        {"query": "qubits in a sustainable photograph of an octopus", "expected": {"Quantum Computing": 3, "Amateur Photography": 2, "Sustainable Architecture": 1}, "category": "chunk_stress", "notes": "4 concepts, k=3 ceiling"},
        {"query": "encrypting data on a mountain trail run", "expected": {"Cybersecurity": 3, "Hiking and Trail Running": 2}, "category": "chunk_stress", "notes": "Unrelated combination"},
        {"query": "telescope photography of black holes for machine learning training data", "expected": {"Astronomy and Astrophysics": 3, "Amateur Photography": 2, "Machine Learning": 1}, "category": "chunk_stress", "notes": "3 distinct technical topics"},

        # ============ TIER 10: Semantic Confusion (Adversarial Adjacency) ============
        # Queries that could plausibly fit multiple topics
        {"query": "neural network training requires lots of compute", "expected": {"Machine Learning": 3}, "category": "semantic_confusion", "notes": "Could leak Quantum"},
        {"query": "encryption keys based on quantum entanglement", "expected": {"Quantum Computing": 3, "Cybersecurity": 2}, "category": "semantic_confusion", "notes": "Genuinely both"},
        {"query": "fermenting coffee cherries before roasting", "expected": {"Specialty Coffee Craft": 3}, "category": "semantic_confusion", "notes": "Could leak Cooking"},
        {"query": "photographing the night sky with a star tracker", "expected": {"Astronomy and Astrophysics": 3, "Amateur Photography": 2}, "category": "semantic_confusion", "notes": "Astrophotography"},
        {"query": "investing in green building real estate", "expected": {"Personal Finance": 3, "Sustainable Architecture": 2}, "category": "semantic_confusion", "notes": "Finance + architecture"},
        {"query": "deep ocean photography expedition gear", "expected": {"Deep Sea Marine Biology": 3, "Amateur Photography": 2}, "category": "semantic_confusion", "notes": "Niche cross-topic"},

        # ============ TIER 11: Conversational Patterns ============
        # Realistic continuation phrases that should defer to context
        {"query": "tell me more about that one", "expected": {}, "category": "conversational", "notes": "Anaphoric, no topic"},
        {"query": "what about the second option", "expected": {}, "category": "conversational", "notes": "Reference, no topic"},
        {"query": "can you elaborate on it", "expected": {}, "category": "conversational", "notes": "Vague reference"},
        {"query": "thanks that was helpful", "expected": {}, "category": "conversational", "notes": "Closing remark"},
        {"query": "wait what does that mean", "expected": {}, "category": "conversational", "notes": "Clarification request"},
        {"query": "good morning Jarvis", "expected": {}, "category": "conversational", "notes": "Greeting with name"},
        
        {
            "query": "torque specifications for a Honda Civic",
            "expected": {},
            "category": "new_topic_cars",
            "notes": "Mechanical specs — no automotive topic exists"
        },
        {
            "query": "best engine oil for high mileage vehicles",
            "expected": {},
            "category": "new_topic_cars",
            "notes": "Maintenance — should reject"
        },
        {
            "query": "Tesla Model 3 versus Toyota Camry comparison",
            "expected": {},
            "category": "new_topic_cars",
            "notes": "Vehicle comparison — should reject"
        },
        {
            "query": "swapping out brake pads on my pickup truck",
            "expected": {},
            "category": "new_topic_cars",
            "notes": "DIY auto repair — should reject"
        },
        
        # --- Watches / Horology ---
        {
            "query": "Rolex Submariner versus Omega Seamaster",
            "expected": {},
            "category": "new_topic_watches",
            "notes": "Luxury watch comparison — should reject"
        },
        {
            "query": "automatic movement versus quartz watch",
            "expected": {},
            "category": "new_topic_watches",
            "notes": "Watch mechanics — should reject"
        },
        {
            "query": "winding a vintage Patek Philippe daily",
            "expected": {},
            "category": "new_topic_watches",
            "notes": "Watch maintenance — should reject"
        },
        
        # --- Fast Food / Quick Service ---
        {
            "query": "McDonald's Quarter Pounder calorie count",
            "expected": {},
            "category": "new_topic_fastfood",
            "notes": "Fast food info — should reject (NOT Home Cooking)"
        },
        {
            "query": "best drive thru breakfast under five dollars",
            "expected": {},
            "category": "new_topic_fastfood",
            "notes": "Drive-thru — should reject"
        },
        {
            "query": "Taco Bell late night menu items",
            "expected": {},
            "category": "new_topic_fastfood",
            "notes": "Chain restaurant — should reject"
        },
        
        # --- Video Games ---
        {
            "query": "Elden Ring boss strategies for Malenia",
            "expected": {},
            "category": "new_topic_gaming",
            "notes": "Video game strategy — should reject"
        },
        {
            "query": "best graphics settings for Cyberpunk 2077",
            "expected": {},
            "category": "new_topic_gaming",
            "notes": "PC gaming config — should reject"
        },
        {
            "query": "speedrunning Super Mario 64 with backwards long jumps",
            "expected": {},
            "category": "new_topic_gaming",
            "notes": "Gaming technique — should reject"
        },
        
        # --- Pets / Animal Care ---
        {
            "query": "litter box training a kitten",
            "expected": {},
            "category": "new_topic_pets",
            "notes": "Pet care — should reject"
        },
        {
            "query": "best dog food brands for senior labrador",
            "expected": {},
            "category": "new_topic_pets",
            "notes": "Pet nutrition — should reject"
        },
        {
            "query": "aquarium water temperature for tropical fish",
            "expected": {},
            "category": "new_topic_pets",
            "notes": "Aquarium care — should reject (NOT Marine Biology)"
        },
        
        # --- Fashion / Style ---
        {
            "query": "best leather boots for winter weather",
            "expected": {},
            "category": "new_topic_fashion",
            "notes": "Footwear shopping — should reject (NOT Hiking)"
        },
        {
            "query": "styling a tailored suit for a wedding",
            "expected": {},
            "category": "new_topic_fashion",
            "notes": "Fashion advice — should reject"
        },
        {
            "query": "vintage denim jacket from the 1980s",
            "expected": {},
            "category": "new_topic_fashion",
            "notes": "Vintage fashion — should reject"
        },
        
        # --- Sports (non-hiking) ---
        {
            "query": "Lebron James career statistics versus Michael Jordan",
            "expected": {},
            "category": "new_topic_sports",
            "notes": "Basketball stats — should reject"
        },
        {
            "query": "Premier League standings this season",
            "expected": {},
            "category": "new_topic_sports",
            "notes": "Soccer — should reject"
        },
        {
            "query": "tennis racket string tension for power players",
            "expected": {},
            "category": "new_topic_sports",
            "notes": "Tennis equipment — should reject"
        },
        
        # --- Travel / Geography ---
        {
            "query": "best neighborhoods to stay in Tokyo for first timers",
            "expected": {},
            "category": "new_topic_travel",
            "notes": "Travel planning — should reject"
        },
        {
            "query": "cheap flights from New York to Lisbon",
            "expected": {},
            "category": "new_topic_travel",
            "notes": "Flight booking — should reject"
        },
        {
            "query": "visa requirements for visiting Brazil",
            "expected": {},
            "category": "new_topic_travel",
            "notes": "Travel docs — should reject"
        },
        
        # --- Movies / TV ---
        {
            "query": "best Christopher Nolan films ranked",
            "expected": {},
            "category": "new_topic_film",
            "notes": "Film criticism — should reject"
        },
        {
            "query": "Breaking Bad versus The Wire which is better",
            "expected": {},
            "category": "new_topic_film",
            "notes": "TV comparison — should reject"
        },
        {
            "query": "Studio Ghibli movies in chronological order",
            "expected": {},
            "category": "new_topic_film",
            "notes": "Animation — should reject"
        },
        
        # --- Gardening / Plants ---
        {
            "query": "growing tomatoes in raised garden beds",
            "expected": {},
            "category": "new_topic_gardening",
            "notes": "Vegetable gardening — should reject (NOT Cooking)"
        },
        {
            "query": "watering schedule for indoor succulents",
            "expected": {},
            "category": "new_topic_gardening",
            "notes": "Houseplant care — should reject"
        },
        {
            "query": "pruning fruit trees in early spring",
            "expected": {},
            "category": "new_topic_gardening",
            "notes": "Orchard care — should reject"
        },
        
        # --- Languages / Linguistics (different angle than existing) ---
        {
            "query": "conjugating French verbs in subjunctive mood",
            "expected": {},
            "category": "new_topic_languages",
            "notes": "Language learning — should reject"
        },
        {
            "query": "Duolingo daily streak strategies",
            "expected": {},
            "category": "new_topic_languages",
            "notes": "Language app — should reject"
        },
        
        # --- Boundary-stress: queries that LOOK like existing topics ---
        # These are designed to be tricky — they share vocabulary but mean different things
        {
            "query": "BMW M3 horsepower and quarter mile times",
            "expected": {},
            "category": "new_topic_cars",
            "notes": "⚠️ 'M3' could weakly hint at Quantum (M3 = matrix?). Should still reject."
        },
        {
            "query": "Apple Watch Ultra battery life review",
            "expected": {},
            "category": "new_topic_watches",
            "notes": "⚠️ 'Watch' might leak. Should reject."
        },
        {
            "query": "Chick-fil-A spicy chicken sandwich recipe copycat",
            "expected": {},
            "category": "new_topic_fastfood",
            "notes": "⚠️ 'Recipe' could leak to Cooking. Borderline acceptable if Cooking only."
        },
        {
            "query": "Nintendo Switch versus Steam Deck for indie games",
            "expected": {},
            "category": "new_topic_gaming",
            "notes": "⚠️ 'Switch' / 'Steam' could hint at tech topics. Should reject."
        },
        {
            "query": "Subaru Outback for camping and trail access",
            "expected": {},
            "category": "new_topic_cars",
            "notes": "⚠️ 'Camping' / 'Trail' might leak to Hiking. Borderline."
        },
    ]
    run_test(tests)
    
    
    
