"""Generate synthetic IAM-style XML handwriting data for training."""

import os
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

# Sample texts that mimic IAM handwriting dataset
SAMPLE_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "a journey of a thousand miles begins with a single step",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "actions speak louder than words",
    "better late than never",
    "curiosity killed the cat",
    "dont count your chickens before they hatch",
    "every cloud has a silver lining",
    "fortune favors the bold",
    "great minds think alike",
    "honesty is the best policy",
    "if it aint broke dont fix it",
    "knowledge is power",
    "let sleeping dogs lie",
    "make hay while the sun shines",
    "never put off till tomorrow what you can do today",
    "once bitten twice shy",
    "practice makes perfect",
    "the pen is mightier than the sword",
    "where there is a will there is a way",
    "you cant judge a book by its cover",
    "a picture is worth a thousand words",
    "birds of a feather flock together",
    "cleanliness is next to godliness",
    "do unto others as you would have them do unto you",
    "early to bed and early to rise makes a man healthy wealthy and wise",
    "give a man a fish and you feed him for a day",
    "hope for the best and prepare for the worst",
    "it takes two to tango",
    "keep your friends close and your enemies closer",
    "look before you leap",
    "many hands make light work",
    "necessity is the mother of invention",
    "out of sight out of mind",
    "people who live in glass houses should not throw stones",
    "rome was not built in a day",
    "slow and steady wins the race",
    "the early bird catches the worm",
    "there is no place like home",
    "two heads are better than one",
    "when in rome do as the romans do",
    "you reap what you sow",
    "a watched pot never boils",
    "brevity is the soul of wit",
    "charity begins at home",
    "dont cry over spilt milk",
    "dont put all your eggs in one basket",
    "easy come easy go",
    "every dog has its day",
]

# Common English words for generating random sentences
WORDS = [
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "our", "their", "what", "which", "who",
    "whom", "whose", "where", "when", "why", "how", "all", "each",
    "every", "both", "few", "many", "much", "some", "any", "no", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "also",
    "now", "here", "there", "then", "once", "never", "always", "often",
    "sometimes", "usually", "generally", "frequently", "commonly",
    "handwriting", "recognition", "neural", "network", "deep", "learning",
    "model", "training", "data", "sequence", "pattern", "generation",
    "text", "word", "letter", "character", "stroke", "pen", "paper",
    "write", "reading", "language", "machine", "computer", "algorithm",
    "system", "process", "method", "approach", "technique", "analysis",
    "research", "study", "experiment", "result", "performance", "accuracy",
    "error", "loss", "function", "optimization", "gradient", "descent",
    "recurrent", "convolutional", "attention", "transformer", "encoder",
    "decoder", "embedding", "representation", "feature", "extraction",
]


def generate_handwriting_strokes(text, num_strokes=None, base_x=100, base_y=200):
    """Generate realistic-looking handwriting strokes for a given text.
    
    Simulates pen movements with natural variations.
    """
    if num_strokes is None:
        # Roughly 2-5 strokes per character
        num_strokes = max(3, len(text) * random.randint(2, 4) // 3)
    
    strokes = []
    x, y = base_x + random.uniform(-20, 20), base_y + random.uniform(-15, 15)
    
    # Word boundaries (pen lifts)
    word_boundaries = set()
    pos = 0
    for word in text.split():
        pos += len(word) + 1  # +1 for space
        word_boundaries.add(pos - 1)
    
    chars_generated = 0
    stroke_idx = 0
    
    for i in range(num_strokes):
        # Determine if this is near a word boundary (pen lift)
        is_word_boundary = chars_generated in word_boundaries or (i > 0 and random.random() < 0.08)
        
        # Generate stroke points (3-8 points per stroke)
        num_points = random.randint(3, 8)
        stroke_points = []
        
        for j in range(num_points):
            # Natural handwriting movement patterns
            if j == 0:
                # First point: small variation from previous end
                if i > 0 and not is_word_boundary:
                    dx = random.uniform(-5, 15)
                    dy = random.uniform(-8, 8)
                elif is_word_boundary:
                    # Jump to next word position
                    dx = random.uniform(30, 80)
                    dy = random.uniform(-20, 20)
                else:
                    dx = random.uniform(0, 10)
                    dy = random.uniform(-5, 5)
            else:
                # Subsequent points: smooth curves
                dx = random.uniform(-3, 25)
                dy = random.uniform(-15, 15)
            
            x += dx
            y += dy
            
            # Add some natural variation and noise
            x += random.gauss(0, 2)
            y += random.gauss(0, 3)
            
            time = i * 100 + j * 10 + random.randint(0, 5)
            stroke_points.append((round(x, 1), round(y, 1), time))
        
        chars_generated += 1
        strokes.append(stroke_points)
    
    return strokes


def create_xml_element(text, strokes, filename):
    """Create an IAM-style XML element tree."""
    root = ET.Element("StrokeSet")
    
    # Add Line element with text attribute
    line_elem = ET.SubElement(root, "Line")
    line_elem.set("text", text)
    
    # Add strokes
    for stroke_points in strokes:
        stroke_elem = ET.SubElement(root, "Stroke")
        for x, y, time in stroke_points:
            point_elem = ET.SubElement(stroke_elem, "Point")
            point_elem.set("x", str(round(x, 1)))
            point_elem.set("y", str(round(y, 1)))
            point_elem.set("time", str(time))
    
    return root


def prettify_xml(elem):
    """Return a pretty-printed XML string."""
    rough_string = ET.tostring(elem, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generate_dataset(output_dir, num_samples=500, seed=42):
    """Generate a synthetic IAM-style XML dataset."""
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic XML files...")
    
    for i in range(num_samples):
        # Pick a text sample or generate a random sentence
        if i < len(SAMPLE_TEXTS):
            text = SAMPLE_TEXTS[i]
        else:
            # Generate random sentence
            num_words = random.randint(3, 12)
            text = " ".join(random.choices(WORDS, k=num_words))
            text = text[0].upper() + text[1:]  # Capitalize first letter
        
        # Generate strokes
        strokes = generate_handwriting_strokes(text)
        
        # Create XML
        filename = f"synthetic_{i:04d}.xml"
        root = create_xml_element(text, strokes, filename)
        xml_str = prettify_xml(root)
        
        # Write file
        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(xml_str)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} files...")
    
    print(f"Done! Generated {num_samples} XML files in {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic IAM-style XML data")
    parser.add_argument("--output_dir", type=str, default="./synthetic_data", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of XML files to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    generate_dataset(args.output_dir, args.num_samples, args.seed)
