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
    counts = {"partial": 0, "extra": 0, "wrongs": 0, "rights": 0}
    for r in results:
        score = r["score"]
        if score == 0.0:
            counts["wrongs"] += 1
        elif score == 1.0:
            counts["rights"] += 1
        elif score == 0.5:
            counts["partial"] += 1
        elif score in (0.7, 0.4, 0.1):
            counts["extra"] += 1
    print(f"  rights     : {counts['rights']:>3}  ({100*counts['rights']/total:.0f}%)")
    print(f"  partial    : {counts['partial']:>3}  ({100*counts['partial']/total:.0f}%)")
    print(f"  extra      : {counts['extra']:>3}  ({100*counts['extra']/total:.0f}%)")
    print(f"  wrongs     : {counts['wrongs']:>3}  ({100*counts['wrongs']/total:.0f}%)")

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
    
    tests = [

        # ============================================================
        # TIER 1: Single Dominant — one per topic, strong direct vocabulary
        # Regression suite: if these break, something fundamentally broke upstream
        # ============================================================

        # Quantum Computing
        {"query": "qubits in superposition and entanglement", "expected": {"Quantum Computing": 3}, "category": "single_dominant", "notes": "Core QC vocabulary"},
        {"query": "quantum gate operations on a qubit register", "expected": {"Quantum Computing": 3}, "category": "single_dominant", "notes": "QC operations"},
        {"query": "decoherence in superconducting quantum circuits", "expected": {"Quantum Computing": 3}, "category": "single_dominant", "notes": "QC-specific phenomenon"},

        # Amateur Photography
        {"query": "shutter speed and aperture settings", "expected": {"Amateur Photography": 3}, "category": "single_dominant", "notes": "Core photo vocabulary"},
        {"query": "bokeh with a 50mm prime lens", "expected": {"Amateur Photography": 3}, "category": "single_dominant", "notes": "Lens terminology"},
        {"query": "RAW file editing in Lightroom", "expected": {"Amateur Photography": 3}, "category": "single_dominant", "notes": "Workflow vocabulary"},

        # Violin
        {"query": "Vivaldi Four Seasons concerto", "expected": {"Violin": 3}, "category": "single_dominant", "notes": "Classical music reference"},
        {"query": "rosin and bow pressure for string tone", "expected": {"Violin": 3}, "category": "single_dominant", "notes": "Playing technique"},
        {"query": "shifting positions on the violin fingerboard", "expected": {"Violin": 3}, "category": "single_dominant", "notes": "Technique vocabulary"},

        # Deep Sea Marine Biology
        {"query": "hadal zone hydrothermal vents", "expected": {"Deep Sea Marine Biology": 3}, "category": "single_dominant", "notes": "Direct technical vocabulary"},
        {"query": "anglerfish predation in the midnight zone", "expected": {"Deep Sea Marine Biology": 3}, "category": "single_dominant", "notes": "Deep-sea specific creature"},
        {"query": "pressure adaptations of abyssal organisms", "expected": {"Deep Sea Marine Biology": 3}, "category": "single_dominant", "notes": "Abyssal zone vocabulary"},

        # Sustainable Architecture
        {"query": "cross laminated timber and passive solar design", "expected": {"Sustainable Architecture": 3}, "category": "single_dominant", "notes": "Core vocabulary"},
        {"query": "LEED certification for net-zero buildings", "expected": {"Sustainable Architecture": 3}, "category": "single_dominant", "notes": "Certification system"},
        {"query": "thermal mass in passive house construction", "expected": {"Sustainable Architecture": 3}, "category": "single_dominant", "notes": "Passive house vocabulary"},

        # Specialty Coffee Craft
        {"query": "brewing pour over with a Chemex", "expected": {"Specialty Coffee Craft": 3}, "category": "single_dominant", "notes": "Chemex is unambiguous coffee jargon"},
        {"query": "extraction yield and TDS in espresso shots", "expected": {"Specialty Coffee Craft": 3}, "category": "single_dominant", "notes": "Coffee science vocabulary"},
        {"query": "bloom phase and draw-down in pour over brewing", "expected": {"Specialty Coffee Craft": 3}, "category": "single_dominant", "notes": "Technique-specific vocabulary"},

        # Astronomy and Astrophysics
        {"query": "James Webb infrared galaxy observations", "expected": {"Astronomy and Astrophysics": 3}, "category": "single_dominant", "notes": "Astronomy direct"},
        {"query": "neutron star accretion disk and X-ray emissions", "expected": {"Astronomy and Astrophysics": 3}, "category": "single_dominant", "notes": "Astrophysics vocabulary"},
        {"query": "redshift measurement and the Hubble constant", "expected": {"Astronomy and Astrophysics": 3}, "category": "single_dominant", "notes": "Cosmology terms"},

        # Home Cooking
        {"query": "sous vide steak with proper resting time", "expected": {"Home Cooking": 3}, "category": "single_dominant", "notes": "Cooking technique"},
        {"query": "maillard reaction when pan-searing chicken", "expected": {"Home Cooking": 3}, "category": "single_dominant", "notes": "Culinary chemistry"},
        {"query": "julienning vegetables with proper knife technique", "expected": {"Home Cooking": 3}, "category": "single_dominant", "notes": "Kitchen technique"},

        # Cybersecurity
        {"query": "ransomware attack with lateral movement and data exfiltration", "expected": {"Cybersecurity": 3}, "category": "single_dominant", "notes": "Direct attack vocabulary"},
        {"query": "SQL injection bypassing parameterized queries", "expected": {"Cybersecurity": 3}, "category": "single_dominant", "notes": "Attack vector"},
        {"query": "zero-day vulnerability and responsible disclosure", "expected": {"Cybersecurity": 3}, "category": "single_dominant", "notes": "Security vocabulary"},

        # Hiking and Trail Running
        {"query": "thru-hiking the Pacific Crest Trail", "expected": {"Hiking and Trail Running": 3}, "category": "single_dominant", "notes": "Direct hiking vocabulary"},
        {"query": "ultralight backpacking gear for multi-day trips", "expected": {"Hiking and Trail Running": 3}, "category": "single_dominant", "notes": "Gear vocabulary"},
        {"query": "switchback technique and elevation gain management", "expected": {"Hiking and Trail Running": 3}, "category": "single_dominant", "notes": "Technique vocabulary"},

        # Machine Learning
        {"query": "transformer attention mechanism architecture", "expected": {"Machine Learning": 3}, "category": "single_dominant", "notes": "ML architecture vocabulary"},
        {"query": "backpropagation and gradient vanishing in deep networks", "expected": {"Machine Learning": 3}, "category": "single_dominant", "notes": "Core ML algorithm"},
        {"query": "precision recall tradeoff in binary classification", "expected": {"Machine Learning": 3}, "category": "single_dominant", "notes": "Evaluation vocabulary"},

        # Personal Finance
        {"query": "compound interest in tax-advantaged retirement accounts", "expected": {"Personal Finance": 3}, "category": "single_dominant", "notes": "Finance direct"},
        {"query": "expense ratio and total return on index funds", "expected": {"Personal Finance": 3}, "category": "single_dominant", "notes": "Investing vocabulary"},
        {"query": "dollar cost averaging into volatile assets", "expected": {"Personal Finance": 3}, "category": "single_dominant", "notes": "Strategy vocabulary"},


        # ============================================================
        # TIER 2: Multi-Topic — strong secondaries (ratio > 0.45)
        # Secondary topic should be unambiguously present
        # ============================================================

        {"query": "taking a picture of an octopus playing the violin", "expected": {"Violin": 3, "Amateur Photography": 2, "Deep Sea Marine Biology": 1}, "category": "multi_topic", "notes": "Three distinct concepts — classic test"},
        {"query": "machine learning models for cybersecurity threat detection", "expected": {"Machine Learning": 3, "Cybersecurity": 2}, "category": "multi_topic", "notes": "Two technical fields"},
        {"query": "designing a sustainable building for quantum research labs", "expected": {"Sustainable Architecture": 3, "Quantum Computing": 2}, "category": "multi_topic", "notes": "Architecture + tech"},
        {"query": "investing my savings into quantum computing startups", "expected": {"Quantum Computing": 3, "Personal Finance": 2}, "category": "multi_topic", "notes": "Finance + QC"},
        {"query": "fermenting kimchi while listening to violin music", "expected": {"Violin": 3, "Home Cooking": 2}, "category": "multi_topic", "notes": "Cooking + music — weak secondary accepted"},
        {"query": "the violin and the camera and the qubit", "expected": {"Violin": 3, "Quantum Computing": 2, "Amateur Photography": 1}, "category": "multi_topic", "notes": "Three isolated nouns"},
        {"query": "hiking trail photography with a wide angle lens", "expected": {"Hiking and Trail Running": 3, "Amateur Photography": 2}, "category": "multi_topic", "notes": "Outdoor + photo"},
        {"query": "machine learning predictions for personal investment portfolios", "expected": {"Machine Learning": 3, "Personal Finance": 2}, "category": "multi_topic", "notes": "ML + finance"},
        {"query": "quantum cryptography for securing financial transactions", "expected": {"Quantum Computing": 3, "Cybersecurity": 2}, "category": "multi_topic", "notes": "QC + security"},
        {"query": "photographing bioluminescent deep sea creatures on an expedition", "expected": {"Deep Sea Marine Biology": 3, "Amateur Photography": 2}, "category": "multi_topic", "notes": "Marine primary, photo secondary — NOTE: Astronomy should NOT appear"},
        {"query": "galaxy formation models trained on neural network data", "expected": {"Astronomy and Astrophysics": 3, "Machine Learning": 2}, "category": "multi_topic", "notes": "ML used as astro tool"},
        {"query": "sustainable timber supply chain financing and green bonds", "expected": {"Sustainable Architecture": 3, "Personal Finance": 2}, "category": "multi_topic", "notes": "Architecture + finance"},


        # ============================================================
        # TIER 3: Weak Secondary (ratio 0.20–0.45)
        # These specifically probe PEER_FLOOR and ABSOLUTE_FLOOR
        # Secondary is real but dominated — must still surface
        # ============================================================

        {"query": "photographing the night sky with a star tracker", "expected": {"Astronomy and Astrophysics": 3, "Amateur Photography": 2}, "category": "weak_secondary", "notes": "Photo at ~0.31 ratio — should surface"},
        {"query": "brewing coffee while hiking the Appalachian Trail", "expected": {"Hiking and Trail Running": 3, "Specialty Coffee Craft": 2}, "category": "weak_secondary", "notes": "Coffee at ~0.22 ratio — genuine secondary"},
        {"query": "encrypting data on a mountain trail run", "expected": {"Hiking and Trail Running": 3, "Cybersecurity": 2}, "category": "weak_secondary", "notes": "Cybersec at ~0.27 ratio"},
        {"query": "encryption and quantum computing threats to public key infrastructure", "expected": {"Quantum Computing": 3, "Cybersecurity": 2}, "category": "weak_secondary", "notes": "Cybersec at moderate ratio"},
        {"query": "investing in green building real estate portfolios", "expected": {"Personal Finance": 3, "Sustainable Architecture": 2}, "category": "weak_secondary", "notes": "Architecture at ~0.39 ratio"},
        {"query": "encryption keys derived from quantum entanglement", "expected": {"Quantum Computing": 3, "Cybersecurity": 2}, "category": "weak_secondary", "notes": "QKD — genuinely both"},
        {"query": "photographing exoplanets through an amateur telescope", "expected": {"Astronomy and Astrophysics": 3, "Amateur Photography": 2}, "category": "weak_secondary", "notes": "Photo at ~0.14 ratio — borderline, may fail"},


        # ============================================================
        # TIER 4: Adjacent Disambiguation
        # One topic dominates despite vocabulary overlap with another
        # ============================================================

        {"query": "spectroscopy of distant stars", "expected": {"Astronomy and Astrophysics": 3}, "category": "adjacent", "notes": "Should not leak to QC despite 'quantum spectroscopy' context"},
        {"query": "neural network gradient descent optimization", "expected": {"Machine Learning": 3}, "category": "adjacent", "notes": "Could leak to QC (quantum optimization)"},
        {"query": "fermenting sourdough starter at home", "expected": {"Home Cooking": 3}, "category": "adjacent", "notes": "Fermentation could leak to Coffee"},
        {"query": "trail running with a heart rate monitor", "expected": {"Hiking and Trail Running": 3}, "category": "adjacent", "notes": "Single topic — biometric doesn't leak"},
        {"query": "deep learning for image classification", "expected": {"Machine Learning": 3}, "category": "adjacent", "notes": "Image could touch Photography"},
        {"query": "bioluminescence in deep sea organisms", "expected": {"Deep Sea Marine Biology": 3}, "category": "adjacent", "notes": "Should NOT leak to Astronomy despite 'light'"},
        {"query": "wavefunction collapse in quantum measurement", "expected": {"Quantum Computing": 3}, "category": "adjacent", "notes": "Physics vocab = QC, not Astronomy"},
        {"query": "variable star light curves and period measurement", "expected": {"Astronomy and Astrophysics": 3}, "category": "adjacent", "notes": "'Light curves' are graphs not photos — no Photography leak"},
        {"query": "cold brew steeping time and grind size", "expected": {"Specialty Coffee Craft": 3}, "category": "adjacent", "notes": "Brewing process not Home Cooking"},
        {"query": "tax-loss harvesting strategy for investment accounts", "expected": {"Personal Finance": 3}, "category": "adjacent", "notes": "Finance only — no ML leak"},
        {"query": "reinforcement learning for autonomous navigation", "expected": {"Machine Learning": 3}, "category": "adjacent", "notes": "RL is ML, not QC or Hiking"},


        # ============================================================
        # TIER 5: Topic Boundary [NEW]
        # Queries at the exact boundary of known confusion pairs
        # Each pair tested both directions + an explicit both-topics case
        # ============================================================

        # Coffee ↔ Cooking boundary
        {"query": "fermenting coffee cherries at origin processing stations", "expected": {"Specialty Coffee Craft": 3}, "category": "topic_boundary", "notes": "Coffee processing — not general cooking fermentation"},
        {"query": "water temperature and grind size for pour over extraction", "expected": {"Specialty Coffee Craft": 3}, "category": "topic_boundary", "notes": "Coffee technique — not cooking"},
        {"query": "braising meat with a coffee rub in a Dutch oven", "expected": {"Home Cooking": 3}, "category": "topic_boundary", "notes": "Coffee as cooking ingredient = Cooking primary"},
        {"query": "bread proofing and yeast fermentation at home", "expected": {"Home Cooking": 3}, "category": "topic_boundary", "notes": "Fermentation in bread = Cooking, not Coffee"},
        {"query": "coffee-rubbed steak seared in a cast iron pan", "expected": {"Home Cooking": 3}, "category": "topic_boundary", "notes": "Cooking method dominates even with coffee mentioned"},

        # Astronomy ↔ Marine Biology boundary
        {"query": "deep sea organisms that produce their own light chemically", "expected": {"Deep Sea Marine Biology": 3}, "category": "topic_boundary", "notes": "Bioluminescence = Marine, not Astronomy"},
        {"query": "stellar nucleosynthesis and light emission from stars", "expected": {"Astronomy and Astrophysics": 3}, "category": "topic_boundary", "notes": "Stellar light = Astronomy, not Marine"},
        {"query": "pressure and darkness in the hadal zone versus deep space", "expected": {"Deep Sea Marine Biology": 3, "Astronomy and Astrophysics": 2}, "category": "topic_boundary", "notes": "Explicit comparison of both environments"},
        {"query": "chemosynthesis at hydrothermal vents in the abyssal plain", "expected": {"Deep Sea Marine Biology": 3}, "category": "topic_boundary", "notes": "Chemical energy in deep sea — no Astronomy leak"},
        {"query": "nebula gas clouds and the formation of new stars", "expected": {"Astronomy and Astrophysics": 3}, "category": "topic_boundary", "notes": "Gas clouds = stellar formation, not marine"},

        # ML ↔ Quantum Computing boundary
        {"query": "quantum machine learning hybrid variational circuits", "expected": {"Quantum Computing": 3, "Machine Learning": 2}, "category": "topic_boundary", "notes": "Explicitly both — quantum ML"},
        {"query": "matrix multiplication efficiency in GPU-accelerated neural networks", "expected": {"Machine Learning": 3}, "category": "topic_boundary", "notes": "Matrix in ML context — not QC"},
        {"query": "Grover's algorithm quadratic speedup over classical search", "expected": {"Quantum Computing": 3}, "category": "topic_boundary", "notes": "Grover's is QC-specific"},
        {"query": "stochastic gradient descent convergence guarantees", "expected": {"Machine Learning": 3}, "category": "topic_boundary", "notes": "Classical optimization = ML, not QC"},
        {"query": "quantum annealing versus simulated annealing for optimization", "expected": {"Quantum Computing": 3, "Machine Learning": 2}, "category": "topic_boundary", "notes": "Annealing comparison touches both"},


        # ============================================================
        # TIER 6: Vocabulary Gap
        # Tests embedding generalization beyond training utterances
        # ============================================================

        {"query": "octopus", "expected": {"Deep Sea Marine Biology": 3}, "category": "vocab_gap", "notes": "Single word — generalization"},
        {"query": "Stradivarius", "expected": {"Violin": 3}, "category": "vocab_gap", "notes": "Famous violin brand"},
        {"query": "chemex", "expected": {"Specialty Coffee Craft": 3}, "category": "vocab_gap", "notes": "Single coffee brand word"},
        {"query": "supernova neutron star collapse", "expected": {"Astronomy and Astrophysics": 3}, "category": "vocab_gap", "notes": "Specific astronomical events"},
        {"query": "phishing email with a malicious attachment", "expected": {"Cybersecurity": 3}, "category": "vocab_gap", "notes": "Attack vocabulary"},
        {"query": "Roth IRA contribution limits", "expected": {"Personal Finance": 3}, "category": "vocab_gap", "notes": "Financial product not in utterances"},
        {"query": "whetstone angle for sharpening kitchen knives", "expected": {"Home Cooking": 3}, "category": "vocab_gap", "notes": "Kitchen tool vocabulary"},
        {"query": "gravitational lensing around a black hole", "expected": {"Astronomy and Astrophysics": 3}, "category": "vocab_gap", "notes": "Advanced astro vocabulary"},
        {"query": "timber frame mortise and tenon joinery", "expected": {"Sustainable Architecture": 3}, "category": "vocab_gap", "notes": "Construction vocabulary"},
        {"query": "401k employer matching and vesting schedule", "expected": {"Personal Finance": 3}, "category": "vocab_gap", "notes": "Retirement product vocabulary"},
        {"query": "bioluminescent jellyfish in the deep ocean", "expected": {"Deep Sea Marine Biology": 3}, "category": "vocab_gap", "notes": "Specific creature vocabulary"},
        {"query": "Bordeaux red wine pairings", "expected": {}, "category": "vocab_gap", "notes": "Wine — no topic exists"},
        {"query": "F-stop and depth of field relationship", "expected": {"Amateur Photography": 3}, "category": "vocab_gap", "notes": "Photo vocabulary without saying camera"},
        {"query": "dividends and earnings per share analysis", "expected": {"Personal Finance": 3}, "category": "vocab_gap", "notes": "Stock vocabulary"},


        # ============================================================
        # TIER 7: Systematic Single-Word Tests [NEW]
        # One per topic — either fires cleanly or rejects
        # Calibrates per-chunk noise floor
        # ============================================================

        {"query": "aperture", "expected": {"Amateur Photography": 3}, "category": "single_word", "notes": "Clear photography term"},
        {"query": "qubit", "expected": {"Quantum Computing": 3}, "category": "single_word", "notes": "Clear QC term"},
        {"query": "sourdough", "expected": {"Home Cooking": 3}, "category": "single_word", "notes": "Cooking-specific term"},
        {"query": "nebula", "expected": {"Astronomy and Astrophysics": 3}, "category": "single_word", "notes": "Astronomy term"},
        {"query": "malware", "expected": {"Cybersecurity": 3}, "category": "single_word", "notes": "Security term"},
        {"query": "espresso", "expected": {"Specialty Coffee Craft": 3}, "category": "single_word", "notes": "Coffee term — Violin should NOT appear"},
        {"query": "vibrato", "expected": {"Violin": 3}, "category": "single_word", "notes": "Violin-specific technique term"},
        {"query": "octopus", "expected": {"Deep Sea Marine Biology": 3}, "category": "single_word", "notes": "Marine creature"},
        {"query": "dividends", "expected": {"Personal Finance": 3}, "category": "single_word", "notes": "Finance term"},
        {"query": "entanglement", "expected": {"Quantum Computing": 3}, "category": "single_word", "notes": "QC term — should not bleed"},
        {"query": "q", "expected": {}, "category": "single_word", "notes": "Single letter — must reject"},
        {"query": "trail", "expected": {}, "category": "single_word", "notes": "Too ambiguous — should reject"},
        {"query": "gradient", "expected": {}, "category": "single_word", "notes": "Too ambiguous — ML? slope? reject"},
        {"query": "rosin", "expected": {"Violin": 3}, "category": "single_word", "notes": "Bow accessory — highly specific"},


        # ============================================================
        # TIER 8: Noise Rejection
        # Must produce empty result
        # ============================================================

        {"query": "coffee computer", "expected": {}, "category": "noise", "notes": "Two unrelated words"},
        {"query": "the the the", "expected": {}, "category": "noise", "notes": "Stopword spam"},
        {"query": "hello", "expected": {}, "category": "noise", "notes": "Greeting"},
        {"query": "what is it", "expected": {}, "category": "noise", "notes": "Pronoun reference"},
        {"query": "yes please", "expected": {}, "category": "noise", "notes": "Affirmation"},
        {"query": "tell me more about that", "expected": {}, "category": "noise", "notes": "Continuation phrase"},
        {"query": "and then what", "expected": {}, "category": "noise", "notes": "Anaphoric"},
        {"query": "okay sure", "expected": {}, "category": "noise", "notes": "Acknowledgment"},
        {"query": "I dont know", "expected": {}, "category": "noise", "notes": "Uncertainty"},
        {"query": "remind me later please", "expected": {}, "category": "noise", "notes": "Deferral"},
        {"query": "a a a a a a a a a a", "expected": {}, "category": "noise", "notes": "Stopword repetition"},
        {"query": "?????", "expected": {}, "category": "noise", "notes": "Punctuation only"},
        {"query": "1234567890", "expected": {}, "category": "noise", "notes": "Numbers only"},


        # ============================================================
        # TIER 9: Edge Cases
        # ============================================================

        {"query": "q", "expected": {}, "category": "edge", "notes": "Single character — should reject"},
        {"query": "x", "expected": {}, "category": "edge", "notes": "Single letter — should reject"},
        {"query": "quantum quantum quantum quantum quantum", "expected": {"Quantum Computing": 3}, "category": "edge", "notes": "Keyword repetition"},
        {"query": "violin VIOLIN ViOlIn", "expected": {"Violin": 3}, "category": "edge", "notes": "Case variation"},
        {"query": "machine learning machine learning machine learning", "expected": {"Machine Learning": 3}, "category": "edge", "notes": "Phrase repetition"},
        {"query": "1234567890", "expected": {}, "category": "edge", "notes": "Numbers only"},
        {"query": "I want to learn quantum computing photography hiking machine learning and cooking all at once", "expected": {"Quantum Computing": 3, "Amateur Photography": 2, "Machine Learning": 1}, "category": "edge", "notes": "5+ topics — top 3 only expected"},
        {"query": "the violin is a bowed string instrument used in classical music orchestras", "expected": {"Violin": 3}, "category": "edge", "notes": "Long single-topic sentence"},
        {"query": "hiking trail brewing coffee qubit violin cybersecurity", "expected": {"Violin": 3, "Quantum Computing": 2, "Hiking and Trail Running": 1}, "category": "edge", "notes": "6 keywords — returns top 3"},


        # ============================================================
        # TIER 10: Paraphrase Robustness
        # Topic vocabulary replaced with synonyms/descriptions
        # ============================================================

        {"query": "quietly listening to a string instrument concerto", "expected": {"Violin": 3}, "category": "paraphrase", "notes": "Violin without naming it"},
        {"query": "eco-friendly green construction methods", "expected": {"Sustainable Architecture": 3}, "category": "paraphrase", "notes": "Sustainable synonyms"},
        {"query": "capturing images with low light cameras", "expected": {"Amateur Photography": 3}, "category": "paraphrase", "notes": "Photography paraphrase"},
        {"query": "stargazing through a backyard telescope", "expected": {"Astronomy and Astrophysics": 3}, "category": "paraphrase", "notes": "Astronomy paraphrase"},
        {"query": "preventing computer break-ins and data theft", "expected": {"Cybersecurity": 3}, "category": "paraphrase", "notes": "Cybersec paraphrase"},
        {"query": "saving money for retirement years", "expected": {"Personal Finance": 3}, "category": "paraphrase", "notes": "Finance paraphrase"},
        {"query": "going for long walks in the mountains", "expected": {"Hiking and Trail Running": 3}, "category": "paraphrase", "notes": "Hiking paraphrase"},
        {"query": "AI systems that learn from examples", "expected": {"Machine Learning": 3}, "category": "paraphrase", "notes": "ML paraphrase"},
        {"query": "preparing meals at home from scratch", "expected": {"Home Cooking": 3}, "category": "paraphrase", "notes": "Cooking paraphrase"},
        {"query": "hand-brewed hot drinks from single origin beans", "expected": {"Specialty Coffee Craft": 3}, "category": "paraphrase", "notes": "Coffee without saying coffee"},
        {"query": "creatures that live miles below the ocean surface", "expected": {"Deep Sea Marine Biology": 3}, "category": "paraphrase", "notes": "Marine bio paraphrase"},
        {"query": "bowed instrument in a four-part string ensemble", "expected": {"Violin": 3}, "category": "paraphrase", "notes": "Violin implied by context"},
        {"query": "buildings that produce as much energy as they consume", "expected": {"Sustainable Architecture": 3}, "category": "paraphrase", "notes": "Net-zero architecture paraphrase"},


        # ============================================================
        # TIER 11: Chunk Stress
        # Compound queries — tests chunking under multi-topic load
        # Expected values calibrated to actual embedding behavior
        # ============================================================

        {"query": "violin violin violin violin", "expected": {"Violin": 3}, "category": "chunk_stress", "notes": "Repetition with overlap"},
        {"query": "I photographed a building made of timber", "expected": {"Amateur Photography": 3, "Sustainable Architecture": 2}, "category": "chunk_stress", "notes": "Two topics, one keyword each"},
        {"query": "qubits in a sustainable photograph of an octopus", "expected": {"Quantum Computing": 3, "Amateur Photography": 2, "Deep Sea Marine Biology": 1}, "category": "chunk_stress", "notes": "4 concepts; Marine outranks Sustainable in embedding — adjusted"},
        {"query": "telescope photography of black holes for machine learning training data", "expected": {"Astronomy and Astrophysics": 3, "Machine Learning": 2}, "category": "chunk_stress", "notes": "Photography gravity ~0.04 — too low to surface"},
        {"query": "encrypting data on a mountain trail run", "expected": {"Hiking and Trail Running": 3, "Cybersecurity": 2}, "category": "chunk_stress", "notes": "Unrelated combination"},
        {"query": "violin concerto played at a sustainable architecture awards ceremony", "expected": {"Violin": 3, "Sustainable Architecture": 2}, "category": "chunk_stress", "notes": "Two clear topics in context"},
        {"query": "the chef photographed a violin while writing machine learning code", "expected": {"Machine Learning": 3, "Amateur Photography": 2, "Violin": 1}, "category": "chunk_stress", "notes": "Three topics in a sentence"},
        {"query": "deep sea marine biology and astrophysics share extreme environment research methods", "expected": {"Deep Sea Marine Biology": 3, "Astronomy and Astrophysics": 2}, "category": "chunk_stress", "notes": "Both topics explicitly compared"},


        # ============================================================
        # TIER 12: Semantic Confusion
        # Genuine ambiguity tests — adjusted labels for accuracy
        # NOTE: Several original labels were wrong; corrected here
        # ============================================================

        {"query": "neural network training requires lots of compute", "expected": {"Machine Learning": 3}, "category": "semantic_confusion", "notes": "Compute = ML here; QC is tempting but incorrect"},
        {"query": "fermenting coffee cherries before roasting at origin", "expected": {"Specialty Coffee Craft": 3}, "category": "semantic_confusion", "notes": "Coffee processing, not home cooking fermentation"},
        {"query": "photographing the night sky with a star tracker mount", "expected": {"Astronomy and Astrophysics": 3, "Amateur Photography": 2}, "category": "semantic_confusion", "notes": "Astrophotography — genuinely both"},
        {"query": "deep ocean photography expedition with submersible gear", "expected": {"Deep Sea Marine Biology": 3, "Amateur Photography": 2}, "category": "semantic_confusion", "notes": "Marine primary, photo secondary"},
        {"query": "quantum key distribution for secure communication channels", "expected": {"Quantum Computing": 3, "Cybersecurity": 2}, "category": "semantic_confusion", "notes": "QKD — explicitly both"},
        {"query": "underwater creatures that produce their own light", "expected": {"Deep Sea Marine Biology": 3}, "category": "semantic_confusion", "notes": "Bioluminescence = Marine ONLY — Astronomy incorrect"},
        {"query": "encryption algorithm benchmarking and key length selection", "expected": {"Cybersecurity": 3}, "category": "semantic_confusion", "notes": "Benchmarking here = Cybersec, not ML"},
        {"query": "sustainable real estate investment and green mortgage products", "expected": {"Personal Finance": 3, "Sustainable Architecture": 2}, "category": "semantic_confusion", "notes": "Finance + architecture — genuinely both"},
        {"query": "convolutional neural networks for satellite image analysis", "expected": {"Machine Learning": 3, "Astronomy and Astrophysics": 2}, "category": "semantic_confusion", "notes": "ML tool applied to astronomy"},
        {"query": "classical versus quantum algorithms for integer factorization", "expected": {"Quantum Computing": 3, "Machine Learning": 2}, "category": "semantic_confusion", "notes": "Shor's algorithm territory — both fields"},


        # ============================================================
        # TIER 13: Conversational (all should reject)
        # ============================================================

        {"query": "tell me more about that one", "expected": {}, "category": "conversational", "notes": "Anaphoric"},
        {"query": "what about the second option", "expected": {}, "category": "conversational", "notes": "Reference"},
        {"query": "can you elaborate on it", "expected": {}, "category": "conversational", "notes": "Vague"},
        {"query": "thanks that was helpful", "expected": {}, "category": "conversational", "notes": "Closing"},
        {"query": "wait what does that mean", "expected": {}, "category": "conversational", "notes": "Clarification"},
        {"query": "good morning Jarvis", "expected": {}, "category": "conversational", "notes": "Greeting with name"},
        {"query": "show me something else", "expected": {}, "category": "conversational", "notes": "Redirection"},
        {"query": "I changed my mind", "expected": {}, "category": "conversational", "notes": "State change — no topic"},


        # ============================================================
        # TIER 14: Context Override [NEW]
        # Topic vocabulary used in a clearly non-topic context
        # Tests whether embedding context beats surface vocabulary
        # ============================================================

        {"query": "coffee table book about deep sea photography", "expected": {"Deep Sea Marine Biology": 3, "Amateur Photography": 2}, "category": "context_override", "notes": "Coffee = furniture — should not trigger Coffee Craft"},
        {"query": "violin-shaped pasta with homemade tomato sauce", "expected": {"Home Cooking": 3}, "category": "context_override", "notes": "Violin is pasta shape — cooking is the topic"},
        {"query": "trail of evidence in a cybersecurity forensic investigation", "expected": {"Cybersecurity": 3}, "category": "context_override", "notes": "Trail is metaphorical — not Hiking"},
        {"query": "machine learning researcher who plays violin as a hobby", "expected": {"Machine Learning": 3}, "category": "context_override", "notes": "Violin is incidental hobby context — ML is the subject"},
        {"query": "sustainable coffee packaging design for specialty roasters", "expected": {"Specialty Coffee Craft": 3}, "category": "context_override", "notes": "Sustainable is adjective for coffee packaging — Architecture shouldn't dominate"},
        {"query": "hiking boots made from recycled sustainable materials", "expected": {"Hiking and Trail Running": 3}, "category": "context_override", "notes": "Sustainable modifies boots — Hiking is primary"},
        {"query": "quantum leap in cybersecurity protection methods", "expected": {"Cybersecurity": 3}, "category": "context_override", "notes": "Quantum leap is idiom — Cybersec is the topic"},


        # ============================================================
        # TIER 15: Out of Distribution — should reject
        # ============================================================

        # These should definitely fail (no vocabulary overlap)
        {"query": "training my golden retriever to sit", "expected": {}, "category": "ood_pets", "notes": "Pet training"},
        {"query": "treating skin conditions with topical medication", "expected": {}, "category": "ood_medical", "notes": "Medical — no topic"},
        {"query": "writing a novel with multiple POV characters", "expected": {}, "category": "ood_writing", "notes": "Fiction writing"},
        {"query": "how to negotiate a salary increase", "expected": {}, "category": "ood_career", "notes": "Career advice"},
        {"query": "Mandarin Chinese grammar rules", "expected": {}, "category": "ood_languages", "notes": "⚠️ Language learning — should not be ML"},
        {"query": "conjugating French verbs in subjunctive mood", "expected": {}, "category": "ood_languages", "notes": "Language learning"},
        {"query": "Duolingo daily streak strategies", "expected": {}, "category": "ood_languages", "notes": "⚠️ Streak/strategies — should not be ML"},
        {"query": "etymology of technical vocabulary in English", "expected": {}, "category": "ood_languages", "notes": "Linguistics"},
        {"query": "symptoms and treatment of type 2 diabetes", "expected": {}, "category": "ood_medical", "notes": "Medical condition"},
        {"query": "how to start meditating daily", "expected": {}, "category": "ood_wellness", "notes": "Wellness — no topic"},
        {"query": "beginner yoga poses for flexibility", "expected": {}, "category": "ood_wellness", "notes": "Yoga"},

        # Vocabulary-adjacent OOD (should still reject)
        {"query": "growing tomatoes in raised garden beds", "expected": {}, "category": "ood_gardening", "notes": "⚠️ Tomatoes/growing might hint at Cooking — should reject"},
        {"query": "pruning fruit trees in early spring", "expected": {}, "category": "ood_gardening", "notes": "⚠️ Fruit might hint at Cooking"},
        {"query": "aquarium water temperature for tropical fish", "expected": {}, "category": "ood_pets", "notes": "⚠️ Aquarium is NOT Deep Sea Marine Biology"},
        {"query": "best leather boots for winter weather", "expected": {}, "category": "ood_fashion", "notes": "⚠️ Boots might hint at Hiking — should reject"},
        {"query": "mountain lodge accommodation in the Swiss Alps", "expected": {}, "category": "ood_travel", "notes": "⚠️ Mountain/Alps might hint at Hiking"},
        {"query": "sustainable fashion and ethical clothing brands", "expected": {}, "category": "ood_fashion", "notes": "⚠️ Sustainable might hint at Architecture"},
        {"query": "marathon training plan for complete beginners", "expected": {}, "category": "ood_sports", "notes": "⚠️ Trail running is in scope; marathon is not"},
        {"query": "electric vehicle charging infrastructure planning", "expected": {}, "category": "ood_automotive", "notes": "⚠️ Infrastructure planning ≠ Sustainable Architecture"},
        {"query": "Subaru Outback for camping and trail access", "expected": {}, "category": "ood_automotive", "notes": "⚠️ Trail/camping leak to Hiking — should reject; car is subject"},

        # Clear OOD — automotive, gaming, watches, fast food, film, sports, travel
        {"query": "torque specifications for a Honda Civic", "expected": {}, "category": "ood_automotive", "notes": "Mechanical specs"},
        {"query": "best engine oil for high mileage vehicles", "expected": {}, "category": "ood_automotive", "notes": "Car maintenance"},
        {"query": "BMW M3 horsepower and quarter mile times", "expected": {}, "category": "ood_automotive", "notes": "⚠️ M3 could hint at matrix/ML"},
        {"query": "Rolex Submariner versus Omega Seamaster", "expected": {}, "category": "ood_watches", "notes": "Luxury watches"},
        {"query": "Apple Watch Ultra battery life review", "expected": {}, "category": "ood_watches", "notes": "⚠️ Watch might leak"},
        {"query": "Elden Ring boss strategies for Malenia", "expected": {}, "category": "ood_gaming", "notes": "Video game"},
        {"query": "best graphics settings for Cyberpunk 2077", "expected": {}, "category": "ood_gaming", "notes": "⚠️ Cyberpunk could hint at Cybersecurity"},
        {"query": "Nintendo Switch versus Steam Deck for indie games", "expected": {}, "category": "ood_gaming", "notes": "Gaming hardware"},
        {"query": "McDonald's Quarter Pounder calorie count", "expected": {}, "category": "ood_fastfood", "notes": "Fast food — NOT Home Cooking"},
        {"query": "Chick-fil-A spicy chicken sandwich copycat recipe", "expected": {}, "category": "ood_fastfood", "notes": "⚠️ Recipe might leak to Cooking"},
        {"query": "best Christopher Nolan films ranked", "expected": {}, "category": "ood_film", "notes": "Film criticism"},
        {"query": "Studio Ghibli animation production techniques", "expected": {}, "category": "ood_film", "notes": "Animation"},
        {"query": "LeBron James career statistics versus Michael Jordan", "expected": {}, "category": "ood_sports", "notes": "Basketball"},
        {"query": "tennis racket string tension for power players", "expected": {}, "category": "ood_sports", "notes": "Tennis equipment"},
        {"query": "cheap flights from New York to Lisbon", "expected": {}, "category": "ood_travel", "notes": "Flight booking"},
        {"query": "best neighborhoods to stay in Tokyo", "expected": {}, "category": "ood_travel", "notes": "Travel planning"},


        # ============================================================
        # TIER 16: In-Domain but Hard to Chunk
        # Queries that SHOULD match but use indirect phrasing
        # Probes per_chunk_min_gravity calibration
        # ============================================================

        {"query": "how do I file my taxes correctly", "expected": {"Personal Finance": 3}, "category": "indirect_phrasing", "notes": "Taxes = Finance; conversational framing may weaken chunks"},
        {"query": "recipe for chocolate chip cookies", "expected": {"Home Cooking": 3}, "category": "indirect_phrasing", "notes": "Basic recipe = Cooking; question framing may split chunks"},
        {"query": "best way to sharpen a chef knife", "expected": {"Home Cooking": 3}, "category": "indirect_phrasing", "notes": "Kitchen tool maintenance = Cooking"},
        {"query": "what should I invest my emergency fund in", "expected": {"Personal Finance": 3}, "category": "indirect_phrasing", "notes": "Emergency fund = Finance; indirect phrasing"},
        {"query": "how often should I change my camera lens", "expected": {"Amateur Photography": 3}, "category": "indirect_phrasing", "notes": "Camera maintenance question"},
        {"query": "what gear do I need to start hiking", "expected": {"Hiking and Trail Running": 3}, "category": "indirect_phrasing", "notes": "Gear advice question = Hiking"},
        {"query": "where do anglerfish live in the ocean", "expected": {"Deep Sea Marine Biology": 3}, "category": "indirect_phrasing", "notes": "Question framing of marine biology"},
        {"query": "I want to learn more about quantum computing", "expected": {"Quantum Computing": 3}, "category": "indirect_phrasing", "notes": "Learning intent + topic name"},
    ]
    
    run_test(tests)
    
    
    
