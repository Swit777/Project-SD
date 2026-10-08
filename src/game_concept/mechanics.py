from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

TAXONOMY_VERSION = "1.0.2"
# Tags and narrow description phrases are evidence of a claim, not verified gameplay quality.
CATALOG = [
    ("crafting", "Crafting", ["Crafting"], [], [r"\bcrafting\b", r"\bcraft (?:your own |new |powerful )?(?:weapons|tools|items|equipment|machines)\b"]),
    ("base_building", "Base building", ["Base Building"], [], [r"\bbase[- ]building\b", r"\bbuild (?:your |a |and defend your )?(?:base|settlement|colony)\b"]),
    ("survival", "Survival systems", ["Survival", "Survival Craft"], [], [r"\bsurvival (?:mechanics|systems|gameplay)\b", r"\b(?:hunger|thirst) (?:and|management|meters)\b"]),
    ("procedural_generation", "Procedural generation", ["Procedural Generation"], [], [r"\bprocedurally[- ]generated\b", r"\bprocedural generation\b"]),
    ("permadeath", "Permadeath", ["Permadeath"], [], [r"\bperma(?:nent[- ]|[- ])?death\b"]),
    ("deck_building", "Deck building", ["Deckbuilding", "Roguelike Deckbuilder"], [], [r"\bdeck[- ]building\b", r"\bbuild (?:your |a )?(?:deck|decks)\b"]),
    ("turn_based_combat", "Turn-based combat", ["Turn-Based Combat", "Turn-Based Tactics"], [], [r"\bturn[- ]based (?:combat|battles|tactics)\b"]),
    ("real_time_combat", "Real-time combat", ["Real-Time with Pause", "Hack and Slash"], [], [r"\breal[- ]time (?:combat|battles)\b", r"\bhack[- ]and[- ]slash\b"]),
    ("skill_tree", "Skill trees", ["Skill Tree"], [], [r"\bskill trees?\b", r"\btalent trees?\b"]),
    ("loot", "Loot systems", ["Loot", "Looter Shooter"], [], [r"\b(?:randomized|random|rare|legendary) loot\b", r"\bloot (?:system|equipment|weapons)\b"]),
    ("character_customization", "Character customization", ["Character Customization"], [], [r"\bcharacter customi[sz]ation\b", r"\bcustomi[sz]e your character\b"]),
    ("branching_story", "Branching narrative", ["Choices Matter", "Multiple Endings", "Choose Your Own Adventure"], [], [r"\bbranching (?:story|narrative|dialogue)\b", r"\bmultiple endings\b", r"\bchoices (?:that )?(?:matter|affect|shape)\b"]),
    ("puzzles", "Puzzles", ["Puzzle", "Puzzle Platformer"], [], [r"\bsolve (?:challenging |intricate |environmental )?puzzles\b", r"\bpuzzle[- ]solving\b"]),
    ("platforming", "Platforming", ["Platformer", "2D Platformer", "3D Platformer", "Precision Platformer"], [], [r"\bplatforming\b", r"\bprecision platformer\b"]),
    ("stealth", "Stealth", ["Stealth"], [], [r"\bstealth (?:mechanics|gameplay|combat|system)\b", r"\bsneak past\b"]),
    ("resource_management", "Resource management", ["Resource Management"], [], [r"\bresource management\b", r"\bmanage (?:your |scarce )?resources\b"]),
    ("farming", "Farming", ["Farming Sim", "Agriculture"], [], [r"\bgrow crops\b", r"\braise animals\b", r"\bfarming (?:system|simulation|mechanics)\b"]),
    ("fishing", "Fishing", ["Fishing"], [], [r"\bfishing (?:mechanics|system|minigame)\b", r"\bcatch (?:different |rare )?fish\b"]),
    ("taming", "Creature taming", [], [], [r"\btame (?:wild |your |and train )?(?:creatures|animals|dinosaurs|monsters)\b", r"\banimal taming\b"]),
    ("trading", "Trading", ["Trading"], [], [r"\btrading (?:system|goods|mechanics)\b", r"\bbuy and sell\b"]),
    ("automation", "Automation", ["Automation"], [], [r"\bautomat(?:e|ed) (?:your |the )?(?:production|factory|factories|systems)\b", r"\bproduction lines\b"]),
    ("physics", "Physics interactions", ["Physics"], [], [r"\bphysics[- ]based (?:puzzles|gameplay|combat|interactions)\b", r"\bphysics simulation\b"]),
    ("parkour", "Parkour", ["Parkour"], [], [r"\bparkour\b", r"\bwall[- ]running\b"]),
    ("rhythm", "Rhythm gameplay", ["Rhythm"], [], [r"\brhythm[- ]based\b", r"\bto the beat\b"]),
    ("time_manipulation", "Time manipulation", ["Time Manipulation"], [], [r"\brewind time\b", r"\btime manipulation\b", r"\bslow down time\b"]),
    ("co_op", "Cooperative play", ["Co-op", "Online Co-Op", "Local Co-Op"], ["Co-op", "Online Co-op", "LAN Co-op", "Shared/Split Screen Co-op"], [r"\bco[- ]?operative (?:play|multiplayer|gameplay)\b", r"\bco[- ]op\b"]),
    ("pvp", "Player versus player", ["PvP"], ["PvP", "Online PvP", "LAN PvP", "Shared/Split Screen PvP"], [r"\bplayer[- ]versus[- ]player\b", r"\bpvp\b"]),
    ("open_world", "Open world", ["Open World"], [], [r"\bopen[- ]world\b"]),
    ("user_content", "User-generated content", ["Moddable", "Level Editor"], ["Steam Workshop", "Includes level editor"], [r"\bmodding support\b", r"\blevel editor\b", r"\buser[- ]generated content\b"]),
    ("tower_defense", "Tower defense", ["Tower Defense"], [], [r"\btower[- ]defen[cs]e\b"]),
    ("exploration", "Exploration", ["Exploration"], [], [r"\bexploration[- ]focused\b", r"\bexplore (?:a |an |the )?(?:vast|open|procedurally)\b"]),
    ("dating", "Relationships / dating", ["Dating Sim", "Romance"], [], [r"\bdating (?:sim|system|mechanics)\b", r"\bromance (?:options|system|characters)\b"]),
    ("quests", "Quests", [], [], [r"\bside quests\b", r"\bcomplete (?:challenging |various )?quests\b", r"\bquest (?:system|lines)\b"]),
    ("boss_battles", "Boss battles", ["Boss Rush"], [], [r"\bboss (?:battles|fights|rush)\b", r"\bfight (?:powerful |epic )?bosses\b"]),
    ("destruction", "Destructible environments", ["Destruction"], [], [r"\bdestructible (?:environments|terrain|objects)\b"]),
    ("vehicle_driving", "Vehicles / driving", ["Driving", "Vehicular Combat"], [], [r"\bdrive (?:cars|vehicles|trucks)\b", r"\bvehicular combat\b"]),
]
KEYS = [entry[0] for entry in CATALOG]
LABELS = {entry[0]: entry[1] for entry in CATALOG}
PATTERNS = {entry[0]: [re.compile(pattern, re.I) for pattern in entry[4]] for entry in CATALOG}


def plain_text(value: str) -> str:
    soup = BeautifulSoup(str(value or ""), "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    return re.sub(r"\s+", " ", html.unescape(soup.get_text(" ", strip=True))).strip()


def is_negated(sentence: str, start: int, end: int) -> bool:
    prefix = sentence[max(0, start - 55):start]
    suffix = sentence[end:end + 35]
    prefix = re.split(r"[;.!?]|,\s*(?:but|however|yet)\b|\b(?:but|however|yet)\b", prefix, flags=re.I)[-1]
    prefix = re.sub(r"\bnot\s+(?:only|just|merely)\b", "", prefix, flags=re.I)
    return bool(re.search(r"\b(?:no|not|without|never|doesn't|don't|isn't|aren't)\b(?:\W+\w+){0,4}\W*$", prefix, re.I)
                or re.match(r"\s+(?:is |are )?(?:not available|not included|not supported)", suffix, re.I))


def extract_feature_claims(description: str, limit: int = 12) -> list[str]:
    """Candidate design claims for human annotation; these are not new model features."""
    soup = BeautifulSoup(str(description or ""), "html.parser")
    bullets = [plain_text(element.get_text(" ", strip=True)) for element in soup.select("li")]
    sentences = re.split(r"(?<=[.!?])\s+|\s+-\s+", plain_text(description))
    action = re.compile(r"\b(?:build|craft|explore|manage|customi[sz]e|upgrade|unlock|choose|fight|solve|tame|trade|create|rewind|automate)\b", re.I)
    result = []
    for sentence in bullets + sentences:
        match = action.search(sentence)
        if match and not is_negated(sentence, match.start(), match.end()) and 30 <= len(sentence) <= 500:
            claim = sentence[:250]
            if claim not in result:
                result.append(claim)
    return result[:limit]


def extract_mechanics(description: str, tags: list[str], categories: list[str], source_url: str,
                      source_kind: str = "steam_snapshot") -> tuple[dict[str, int], list[dict]]:
    tag_set, category_set = {str(t).casefold() for t in tags}, {str(c).casefold() for c in categories}
    text = plain_text(description)
    sentences = re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
    evidence = []
    features = {}
    for key, label, aliases, category_aliases, _ in CATALOG:
        positive = False
        for kind, options, values in [("tag", aliases, tag_set), ("category", category_aliases, category_set)]:
            for alias in options:
                if alias.casefold() in values:
                    evidence.append({"mechanic": key, "label": label, "source_type": kind,
                                     "evidence": alias, "negated": False, "source_url": source_url, "source_kind": source_kind})
                    positive = True
        seen_polarities = set()
        for sentence in sentences:
            for pattern in PATTERNS[key]:
                for match in pattern.finditer(sentence):
                    negated = is_negated(sentence, match.start(), match.end())
                    if negated in seen_polarities:
                        continue
                    evidence.append({"mechanic": key, "label": label, "source_type": "description_claim",
                        "evidence": sentence[max(0, match.start() - 35):match.end() + 70][:180],
                        "matched_phrase": match.group(), "negated": negated, "source_url": source_url, "source_kind": source_kind})
                    positive |= not negated
                    seen_polarities.add(negated)
        features[f"mechanic_{key}"] = int(positive)
    return features, evidence
