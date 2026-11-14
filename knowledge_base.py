"""Rule-based mapping: model class -> agronomy findings and action plan.

Keys must match your dataset folder names (lower-case, underscores).
To add a class (e.g. magnesium_deficiency), train with that folder present and
add one more entry below; unknown classes fall back to GENERIC.

NOTE: product rates are intentionally NOT hard-coded. Always follow the label
of the specific fertilizer and confirm with a soil/tissue test.
"""

KNOWLEDGE_BASE = {
    "healthy": {
        "title": "Healthy plant",
        "category": "healthy",
        "severity": "none",
        "summary": "No visible stress pattern detected. Foliage colour and structure look normal.",
        "symptoms": ["Uniform green foliage", "Turgid leaves", "No lesions or scorching"],
        "root_causes": [],
        "soil_amendments": ["No amendment needed. Maintain current feeding programme."],
        "precautions": [
            "Keep scouting weekly; early symptoms start on older or newest leaves.",
            "Photograph the plant on a regular schedule to track change.",
        ],
        "prevention": [
            "Soil-test once or twice a year and keep pH in the crop's preferred range.",
            "Water at the base, mulch to hold moisture, rotate crops each season.",
        ],
    },
    "nitrogen_deficiency": {
        "title": "Nitrogen (N) deficiency",
        "category": "nutrient",
        "severity": "moderate",
        "summary": "Pattern consistent with nitrogen shortage: general yellowing that starts on "
                   "the OLDER, lower leaves because N is mobile and moves to new growth.",
        "symptoms": ["Uniform pale green to yellow older leaves (chlorosis)",
                     "Stunted, spindly growth", "Reduced leaf size and early leaf drop"],
        "root_causes": ["Low soil organic matter or leached soil (heavy rain / over-watering)",
                        "Heavy feeders in nutrient-depleted beds",
                        "High-carbon mulch or wood chips tying up N (immobilisation)"],
        "soil_amendments": [
            "Quick fix: water-soluble N or a diluted liquid feed (fish emulsion, seaweed blend) per label.",
            "Slow release: well-composted manure, blood meal, or a balanced slow-release fertilizer.",
            "Long term: add compost and grow legume cover crops / green manure.",
        ],
        "precautions": [
            "Do not over-apply: excess N gives soft growth, attracts pests and delays fruiting.",
            "Split applications into small doses rather than one heavy dose.",
            "Rule out root problems or waterlogging, which mimic N deficiency.",
        ],
        "prevention": ["Annual compost top-dressing", "Avoid over-irrigation and nutrient leaching",
                       "Rotate heavy feeders with legumes"],
    },
    "phosphorus_deficiency": {
        "title": "Phosphorus (P) deficiency",
        "category": "nutrient",
        "severity": "moderate",
        "summary": "Pattern consistent with phosphorus shortage: dark green or purplish/bronze "
                   "older leaves and slow, stunted growth, often in cold or acidic/alkaline soils.",
        "symptoms": ["Dark green, purple or reddish tint on older leaves and stems",
                     "Slow growth and delayed flowering", "Poor root development"],
        "root_causes": ["Cold soil (P uptake drops sharply at low temperature)",
                        "Soil pH too low (<5.5) or too high (>7.5), which locks P up",
                        "Low available P in soil; compacted or waterlogged roots"],
        "soil_amendments": [
            "Bone meal or composted manure for organic P; superphosphate per label.",
            "Rock phosphate works slowly and only in acidic soil.",
            "Correct pH first (lime if acidic, sulfur if alkaline) so existing P becomes available.",
        ],
        "precautions": [
            "Check soil temperature; purpling in cold spells may resolve by itself.",
            "Avoid excess P: it can induce zinc and iron deficiency and runs off into waterways.",
            "Apply near the root zone; P barely moves in soil.",
        ],
        "prevention": ["Maintain pH 6.0-7.0 for most crops", "Use mycorrhizal inoculants at planting",
                       "Warm the soil early in season (mulch film, raised beds)"],
    },
    "potassium_deficiency": {
        "title": "Potassium (K) deficiency",
        "category": "nutrient",
        "severity": "moderate to high",
        "summary": "Pattern consistent with potassium shortage: yellowing and scorched, brown "
                   "margins and tips on older leaves, weak stems and poor fruit quality.",
        "symptoms": ["Yellow then brown leaf edges and tips (marginal scorch) on older leaves",
                     "Weak stems and lodging", "Poor fruit set, uneven ripening"],
        "root_causes": ["Sandy soils with low K-holding capacity", "Excess calcium or magnesium competing with K",
                        "Drought stress or heavy fruiting load"],
        "soil_amendments": [
            "Potassium sulfate (sulfate of potash) per label; use sulfate form for chloride-sensitive crops.",
            "Organic options: compost, kelp meal, greensand (slow).",
            "Wood ash supplies K but raises pH sharply; use sparingly and only on acidic soil.",
        ],
        "precautions": [
            "Over-application of K can induce magnesium and calcium deficiency.",
            "Confirm with a tissue test; salt injury and drought produce similar scorch.",
            "Keep soil evenly moist so K can be taken up.",
        ],
        "prevention": ["Regular soil tests, particularly on sandy soils", "Mulch to conserve moisture",
                       "Balanced N-P-K feeding, especially during fruiting"],
    },
    "dead_necrotic": {
        "title": "Dead / necrotic specimen (post-mortem)",
        "category": "dead",
        "severity": "critical",
        "summary": "Tissue appears largely dead (brown/black, desiccated or collapsed). A photo cannot "
                   "by itself prove WHY the plant died, so use the post-mortem checklist to find the cause.",
        "symptoms": ["Widespread brown/black or crispy tissue", "Collapsed stems", "No green tissue"],
        "root_causes": [
            "Root rot / waterlogging (mushy, dark, foul-smelling roots)",
            "Prolonged drought or heat scorch (dry, brittle roots and crispy leaves)",
            "Frost or freeze damage (sudden blackening, water-soaked tissue)",
            "Vascular wilt, stem canker or other disease (brown streaking inside stem)",
            "Pests (borers, root grubs, nematodes) or herbicide drift / fertilizer burn",
            "Severe long-term nutrient deficiency or toxicity",
        ],
        "soil_amendments": [
            "Do not re-fertilise a dead plant; test the soil (pH, EC/salinity, N-P-K) before replanting.",
            "If root rot or disease is suspected: improve drainage, add compost, or solarise the soil.",
            "If salts/fertilizer burn is suspected: flush the bed with plenty of water.",
        ],
        "precautions": [
            "Scratch test: scrape a stem; green tissue underneath means the plant may still recover. "
            "Cut back dead growth and keep watering sensibly.",
            "Inspect roots: firm and white = healthy; brown, slimy and smelly = rot.",
            "Remove diseased plants and dispose of them (bin, not compost).",
            "Disinfect tools and pots (10% bleach or 70% alcohol).",
            "Isolate neighbouring plants and check them for the same symptoms.",
        ],
        "prevention": ["Match watering to soil drying, not the calendar", "Use well-drained soil and clean pots",
                       "Protect from frost; harden off seedlings", "Rotate crops; do not replant the same family in diseased soil"],
    },
}

GENERIC = {
    "title": "Unrecognised class",
    "category": "unknown",
    "severity": "unknown",
    "summary": "No specific guidance is stored for this class. Add an entry to knowledge_base.py.",
    "symptoms": [], "root_causes": [], "soil_amendments": [],
    "precautions": ["Confirm with a soil test and an agronomist."], "prevention": [],
}


def normalise(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def get_recommendations(label: str) -> dict:
    return KNOWLEDGE_BASE.get(normalise(label), GENERIC)
