"""Truth experiment, stage 1 prompts.
Two statement sets (Marks & Tegmark-style 'geometry of truth', rebuilt synthetically):
  CITIES: "The city of {city} is in {country}." for 60 well-known cities; TRUE with the correct country, FALSE with a
          wrong country taken from a seeded derangement of the true-country multiset (every country appears equally often
          in true and false statements; revised after Astra's review); each city once true, once false (120 statements).
  COMPARE: "{a} is larger than {b}." for 60 random two-digit pairs (|a-b| in 1..40), each pair in both orders (one true,
           one false; 120 statements). Knowledge-free; magnitude only.
Two formats per statement:
  A (bare): a fixed unlabeled prefix of four unrelated statements (two true, two false), then the statement; the state of
    interest is at the statement's final "." token.
  B (judgment): three labeled few-shot examples (unrelated facts) "Statement: ...\nTrue or false? True|False\n", then
    "Statement: {s}\nTrue or false?"; answer read as " True" vs " False" at the last token; the statement-final "." token
    position is recorded too.
Held-out unit: city (both statements) / number pair (both orders).
Usage: python -I huginn_addition/make_truth_prompts.py results/truth_prompts.json
"""
import json
import random
import sys

CITIES = [("Paris", "France"), ("Tokyo", "Japan"), ("Berlin", "Germany"), ("Madrid", "Spain"), ("Rome", "Italy"), ("Cairo", "Egypt"),
          ("Moscow", "Russia"), ("Beijing", "China"), ("Toronto", "Canada"), ("Sydney", "Australia"), ("Mumbai", "India"),
          ("Lagos", "Nigeria"), ("Nairobi", "Kenya"), ("Lima", "Peru"), ("Bogota", "Colombia"), ("Santiago", "Chile"),
          ("Buenos Aires", "Argentina"), ("Mexico City", "Mexico"), ("Havana", "Cuba"), ("Dublin", "Ireland"), ("Lisbon", "Portugal"),
          ("Athens", "Greece"), ("Istanbul", "Turkey"), ("Tehran", "Iran"), ("Baghdad", "Iraq"), ("Bangkok", "Thailand"),
          ("Hanoi", "Vietnam"), ("Seoul", "South Korea"), ("Manila", "the Philippines"), ("Jakarta", "Indonesia"), ("Oslo", "Norway"),
          ("Stockholm", "Sweden"), ("Helsinki", "Finland"), ("Copenhagen", "Denmark"), ("Warsaw", "Poland"), ("Prague", "the Czech Republic"),
          ("Vienna", "Austria"), ("Budapest", "Hungary"), ("Amsterdam", "the Netherlands"), ("Brussels", "Belgium"), ("Zurich", "Switzerland"),
          ("Chicago", "the United States"), ("Rio de Janeiro", "Brazil"), ("Cape Town", "South Africa"), ("Casablanca", "Morocco"),
          ("Accra", "Ghana"), ("Addis Ababa", "Ethiopia"), ("Karachi", "Pakistan"), ("Dhaka", "Bangladesh"), ("Kabul", "Afghanistan"),
          ("Riyadh", "Saudi Arabia"), ("Kyiv", "Ukraine"), ("Bucharest", "Romania"), ("Belgrade", "Serbia"), ("Melbourne", "Australia"),
          ("Osaka", "Japan"), ("Shanghai", "China"), ("Barcelona", "Spain"), ("Milan", "Italy"), ("Hamburg", "Germany")]
PREFIX_A = ("Water boils at 100 degrees Celsius at sea level.\nThe moon is made of cheese.\n"
            "Cats are mammals.\nThe sun rises in the west.\n")
SHOTS_B = ("Statement: The Pacific is an ocean.\nTrue or false? True\n"
           "Statement: Spiders have six legs.\nTrue or false? False\n"
           "Statement: Ice is frozen water.\nTrue or false? True\n")


def main(out):
    rng = random.Random(20261108)
    true_c = [c for _, c in CITIES]
    while True:  # wrong countries = a seeded permutation of the true-country multiset with no fixed points (balanced marginals)
        perm = true_c[:]; rng.shuffle(perm)
        if all(p != t for p, t in zip(perm, true_c)):
            break
    rows = []
    for i, (city, ctry) in enumerate(CITIES):
        wrong = perm[i]
        for truth, c in ((1, ctry), (0, wrong)):
            rows.append(dict(dataset="cities", unit=f"city{i}", truth=truth, statement=f"The city of {city} is in {c}."))
    pairs = set()
    while len(pairs) < 60:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if 1 <= abs(a - b) <= 40 and (min(a, b), max(a, b)) not in pairs:
            pairs.add((min(a, b), max(a, b)))
    for j, (lo, hi) in enumerate(sorted(pairs)):
        rows.append(dict(dataset="compare", unit=f"pair{j}", truth=1, statement=f"{hi} is larger than {lo}."))
        rows.append(dict(dataset="compare", unit=f"pair{j}", truth=0, statement=f"{lo} is larger than {hi}."))
    for k, r in enumerate(rows):
        r["uid"] = 100000 + k
        r["prompt_A"] = PREFIX_A + r["statement"]
        r["prompt_B"] = SHOTS_B + f"Statement: {r['statement']}\nTrue or false?"
    json.dump(rows, open(out, "w"), indent=1)
    print(len(rows), "statements;", sum(r["truth"] for r in rows), "true"); print(rows[1]["prompt_B"])


if __name__ == "__main__":
    main(sys.argv[1])
