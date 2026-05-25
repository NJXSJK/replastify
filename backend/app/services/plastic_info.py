# backend/app/services/plastic_info.py

PLASTIC_DATABASE: dict[str, dict] = {
    "HDPE": {
        "full_name": "High-Density Polyethylene",
        "resin_code": 2,
        "common_uses": ["Milk jugs", "Detergent bottles", "Shampoo bottles", "Plastic lumber", "Grocery bags"],
        "recyclability": "Widely recycled",
        "recyclability_score": 5,
        "health_concerns": "Considered one of the safest plastics. Non-toxic, does not leach chemicals under normal use.",
        "decomposition_years": 500,
        "recycling_tips": [
            "Rinse containers before placing in recycling",
            "Remove pumps and sprayers — they are usually a different plastic",
            "Check local curbside guidelines; HDPE is widely accepted",
        ],
        "reuse_ideas": ["Garden planters", "Toy storage bins", "Workshop parts organiser"],
        "eco_alternatives": ["Glass jars", "Stainless steel containers", "Bamboo dispensers"],
        "fun_fact": "Recycled HDPE is used to make park benches, playground equipment, and plastic lumber.",
        "warning": None,
    },
    "LDPE": {
        "full_name": "Low-Density Polyethylene",
        "resin_code": 4,
        "common_uses": ["Grocery bags", "Bread bags", "Cling wrap", "Squeezable bottles", "Six-pack rings"],
        "recyclability": "Limited — not typically curbside; grocery store drop-off required",
        "recyclability_score": 3,
        "health_concerns": "Generally considered safe. Does not leach harmful chemicals under normal use.",
        "decomposition_years": 500,
        "recycling_tips": [
            "Return to grocery store drop-off bins (not curbside)",
            "Bundle multiple bags together before dropping off",
            "Keep dry — wet bags are rejected at facilities",
        ],
        "reuse_ideas": ["Bin liners", "Packing cushioning", "Waterproofing layer in garden beds"],
        "eco_alternatives": ["Canvas tote bags", "Silicone food bags", "Beeswax wraps"],
        "fun_fact": "Some grocery store drop-off programs recycle LDPE bags into composite decking material.",
        "warning": None,
    },
    "PET": {
        "full_name": "Polyethylene Terephthalate",
        "resin_code": 1,
        "common_uses": ["Water bottles", "Soda bottles", "Food jars", "Salad dressing containers", "Polyester clothing"],
        "recyclability": "Widely recycled",
        "recyclability_score": 5,
        "health_concerns": "Safe for single use. Prolonged reuse may harbour bacteria in micro-scratches. Can leach antimony if heated.",
        "decomposition_years": 450,
        "recycling_tips": [
            "Rinse thoroughly before recycling",
            "Remove bottle caps — they are PP (#5), not PET",
            "Flatten to save space in the recycling bin",
        ],
        "reuse_ideas": ["Plant pots", "Bird feeders", "Piggy banks"],
        "eco_alternatives": ["Glass bottles", "Stainless steel bottles", "Aluminium cans"],
        "fun_fact": "Recycling one PET bottle saves enough energy to power a laptop for approximately 25 minutes.",
        "warning": None,
    },
    "PP": {
        "full_name": "Polypropylene",
        "resin_code": 5,
        "common_uses": ["Yogurt tubs", "Bottle caps", "Straws", "Microwave-safe containers", "Medicine bottles", "Tupperware"],
        "recyclability": "Increasingly accepted in curbside programmes — check locally",
        "recyclability_score": 4,
        "health_concerns": "Considered safe. Heat-resistant — one of the few plastics suitable for microwave use. Does not leach chemicals at normal temperatures.",
        "decomposition_years": 20,
        "recycling_tips": [
            "Check your local municipal guidelines — acceptance varies widely",
            "Clean thoroughly; food residue causes rejection at facilities",
            "Separate lids from bottles before recycling",
        ],
        "reuse_ideas": ["Food storage containers", "Seed-starting trays", "Workshop organiser bins"],
        "eco_alternatives": ["Glass jars", "Stainless steel containers", "Bamboo straws"],
        "fun_fact": "PP has one of the shortest decomposition times of common plastics — just 20–30 years vs 450+ for PET.",
        "warning": None,
    },
    "PS": {
        "full_name": "Polystyrene (Styrofoam)",
        "resin_code": 6,
        "common_uses": ["Foam cups", "Takeout containers", "Packing peanuts", "CD cases", "Disposable plates"],
        "recyclability": "Difficult — rarely accepted curbside",
        "recyclability_score": 1,
        "health_concerns": "Can leach styrene (a possible carcinogen), especially when heated. Avoid microwaving food in PS containers.",
        "decomposition_years": 500,
        "recycling_tips": [
            "Do not place in general recycling — it breaks apart and contaminates other batches",
            "Search for specialist PS drop-off or mail-back facilities",
            "Reuse foam packing material for shipping fragile items",
        ],
        "reuse_ideas": ["Packing material for shipping", "Insulation in cold frames", "Craft project bases"],
        "eco_alternatives": ["Paper/fibre cups", "Compostable PLA containers", "Reusable stainless steel mugs"],
        "fun_fact": "Polystyrene is 95% air by volume — incredibly light, but this makes it almost impossible to recycle economically.",
        "warning": "Avoid heating food in PS containers — styrene may migrate into food.",
    },
    "PVC": {
        "full_name": "Polyvinyl Chloride",
        "resin_code": 3,
        "common_uses": ["Pipes", "Shower curtains", "Cling wrap", "Vinyl flooring", "Window frames", "Cable insulation"],
        "recyclability": "Difficult — specialist facilities only",
        "recyclability_score": 2,
        "health_concerns": "Contains harmful phthalates and chlorine. Can release toxic dioxins when burned. Avoid for food or drink contact.",
        "decomposition_years": 1000,
        "recycling_tips": [
            "Never place in general recycling — it contaminates entire batches",
            "Contact specialist PVC recyclers or manufacturer take-back programmes",
            "If renovation material, check if local councils have specific drop-off points",
        ],
        "reuse_ideas": ["Pipe sections as garden edging", "PVC sheet as workshop surface protector"],
        "eco_alternatives": ["Copper or PEX pipes", "Silicone wraps", "Natural fibre shower curtains"],
        "fun_fact": "PVC is one of the most widely produced plastics globally, yet one of the hardest to safely recycle or dispose of.",
        "warning": "Contains chlorine and phthalates. Never burn PVC — releases highly toxic dioxins.",
    },
}


def get_plastic_info(class_name: str) -> dict:
    """
    Return the full info dict for a plastic type.
    Raises KeyError with a clear message if class_name is not recognised.
    """
    if class_name not in PLASTIC_DATABASE:
        valid = list(PLASTIC_DATABASE.keys())
        raise KeyError(f"Unknown plastic type: '{class_name}'. Valid types: {valid}")
    return PLASTIC_DATABASE[class_name]
