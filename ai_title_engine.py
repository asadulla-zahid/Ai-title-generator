# ai_title_engine.py
#
# AI-powered Title & Location Analyzer using pretrained Transformers.
# - Title generation: t5-small (light, decent quality)
# - Location extraction: dslim/bert-base-NER (NER for locations)
# - No NLTK (regex tokenizer only)

import re
import heapq
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def simple_tokenize(text: str):
    """Lightweight tokenizer (no NLTK needed)."""
    return re.findall(r"\b\w+\b", text.lower())


class AiTextEngine:
    """
    Provides:
      - generate_titles(paragraph, num_titles=5)
      - extract_locations(paragraph)
    """

    def __init__(self, title_model_name: str = "t5-small", ner_model_name: str = "dslim/bert-base-NER"):
        print(f"[INIT] Loading title model '{title_model_name}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Title model
        self.title_tokenizer = AutoTokenizer.from_pretrained(title_model_name)
        self.title_model = AutoModelForSeq2SeqLM.from_pretrained(title_model_name).to(self.device)
        print(f"[INIT] Title model loaded on {self.device}.")

        # NER model for locations
        print(f"[INIT] Loading NER model '{ner_model_name}' for location parsing...")
        # Use CPU for NER for simplicity / compatibility
        self.ner = pipeline(
            "token-classification",
            model=ner_model_name,
            aggregation_strategy="simple",
            device=-1,
        )
        print("[INIT] NER model loaded.")

    # -------------------------------------------------------------
    # Title generation helpers
    # -------------------------------------------------------------
    def _preprocess_for_title(self, paragraph: str) -> str:
        cleaned = re.sub(r"\s+", " ", paragraph).strip()

        if len(cleaned) > 2000:
            cleaned = cleaned[:2000]

        # T5 summarization-style prefix
        return "summarize: " + cleaned

    def _generate_raw_title_candidates(self, paragraph: str, num_candidates: int = 5):
        if not paragraph.strip():
            return []

        processed = self._preprocess_for_title(paragraph)

        inputs = self.title_tokenizer.encode(
            processed,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        num_candidates = max(1, num_candidates)

        outputs = self.title_model.generate(
            inputs,
            max_new_tokens=64,   # allow a full sentence
            do_sample=True,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=num_candidates,
            repetition_penalty=1.8,
            no_repeat_ngram_size=3,
        )

        return [
            self.title_tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

    def _clean_title(self, title: str) -> str:
        """
        Clean raw model output into a full sentence title.
        """
        title = title.strip()
        title = re.sub(r'^[\'"]+|[\'"]+$', "", title)

        # Keep up to the first full stop
        dot_index = title.find(".")
        if dot_index != -1:
            title = title[:dot_index + 1].strip()
        else:
            # If no period, keep as-is and add one
            title = title.strip()
            if title and not title.endswith("."):
                title += "."

        # Normalize repeated periods
        title = re.sub(r"\.{2,}$", ".", title)

        # Normalize whitespace
        title = re.sub(r"\s+", " ", title).strip()

        # Capitalize first letter only
        if title:
            title = title[0].upper() + title[1:]

        return title

    def _score_title(self, title: str, paragraph: str) -> float:
        if not title:
            return float("-inf")

        paragraph_words = simple_tokenize(paragraph)
        title_words = simple_tokenize(title)

        # Important words = longer words in paragraph
        important = [w for w in paragraph_words if len(w) > 4]

        overlap = len(set(title_words) & set(important))

        # Ideal length ≈ 12 words
        length_penalty = abs(len(title_words) - 12)

        return overlap * 2 - length_penalty

    # -------------------------------------------------------------
    # Public: generate_titles
    # -------------------------------------------------------------
    def generate_titles(self, paragraph: str, num_titles: int = 5):
        paragraph = paragraph.strip()
        if not paragraph:
            return []

        # Generate more than needed
        raw_candidates = self._generate_raw_title_candidates(paragraph, num_titles * 2)

        cleaned = [self._clean_title(t) for t in raw_candidates]

        # Remove duplicates
        seen = set()
        unique = []
        for t in cleaned:
            if t and t not in seen:
                seen.add(t)
                unique.append(t)

        if not unique:
            return []

        scored = [(self._score_title(t, paragraph), t) for t in unique]

        # Select best N
        top = heapq.nlargest(num_titles, scored, key=lambda x: x[0])

        return [t for _, t in top]

    # -------------------------------------------------------------
    # Public: extract_locations
    # -------------------------------------------------------------
    def extract_locations(self, paragraph: str):
        """
        Extract locations (cities, countries, places) and address-like patterns
        from the paragraph using a NER model + regex heuristics.
        Returns a list of unique strings.
        """
        paragraph = paragraph.strip()
        if not paragraph:
            return []

        locations = set()

        # 1) NER-based location detection
        try:
            entities = self.ner(paragraph)
            for ent in entities:
                group = ent.get("entity_group") or ent.get("entity")
                if group and "LOC" in group:
                    text = ent.get("word") or ent.get("entity")
                    if text:
                        locations.add(text.strip())
        except Exception as e:
            print(f"[WARN] NER failed: {e}")

        # 2) Regex-based address patterns (simple heuristic)
        address_pattern = re.compile(
            r"\b\d{1,5}\s+[A-Za-z0-9\.\s]+?"
            r"(Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|"
            r"Lane|Ln\.|Drive|Dr\.|Court|Ct\.|Place|Pl\.)\b",
            re.IGNORECASE,
        )

        for match in address_pattern.finditer(paragraph):
            loc = match.group(0).strip()
            if loc:
                locations.add(loc)

        return sorted(locations)
