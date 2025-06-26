from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List, Dict, Set
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter


class WordCloudGenerator:
    """Generate intelligent word clouds from posts using advanced NLP filtering."""

    def __init__(self) -> None:
        """Initialize word cloud generator with comprehensive filtering."""
        # Download required NLTK data
        for resource in ["stopwords", "punkt", "averaged_perceptron_tagger"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" 
                             else f"taggers/{resource}" if resource == "averaged_perceptron_tagger"
                             else f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        # Comprehensive stopwords including common verbs and auxiliary words
        self.stopwords = set(stopwords.words("english")) | {
            # Basic articles, prepositions, conjunctions
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "up", "about", "into", "through", "during", "before", "after", "above", "below", "between", "under",
            
            # Common auxiliary and modal verbs
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "can", "may", "might", "must", "shall", "ought",
            
            # Common verbs that add little meaning
            "make", "made", "making", "get", "got", "getting", "go", "going", "went", "come", "came", "coming",
            "take", "took", "taking", "see", "saw", "seeing", "know", "knew", "knowing", "think", "thought", "thinking",
            "say", "said", "saying", "tell", "told", "telling", "give", "gave", "giving", "find", "found", "finding",
            "look", "looked", "looking", "use", "used", "using", "work", "worked", "working", "try", "tried", "trying",
            "seem", "seemed", "seeming", "become", "became", "becoming", "want", "wanted", "wanting", "need", "needed", "needing",
            
            # Pronouns and determiners
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "her", "its",
            "our", "their", "mine", "yours", "hers", "ours", "theirs", "this", "that", "these", "those", "some", "any",
            "all", "each", "every", "no", "none", "both", "either", "neither", "much", "many", "more", "most", "few", "little",
            
            # Common adverbs
            "really", "very", "so", "too", "quite", "rather", "pretty", "just", "only", "even", "also", "still", "yet",
            "already", "always", "never", "sometimes", "often", "usually", "maybe", "perhaps", "probably", "definitely",
            "certainly", "surely", "clearly", "obviously", "actually", "basically", "generally", "specifically",
            
            # Time and frequency
            "now", "then", "today", "yesterday", "tomorrow", "here", "there", "where", "when", "how", "why", "what", "which", "who",
            "again", "back", "away", "around", "down", "over", "off", "out", "up", "through", "against", "across", "along",
            
            # Internet/Social media specific
            "rt", "via", "amp", "https", "http", "www", "com", "org", "net", "like", "share", "comment", "post", "thread",
            "reddit", "twitter", "facebook", "instagram", "youtube", "tiktok", "link", "url", "edit", "delete", "update",
            
            # Reddit specific
            "subreddit", "upvote", "downvote", "karma", "mod", "moderator", "op", "original", "poster", "submission",
            
            # Generic expressions
            "lol", "lmao", "haha", "yeah", "yes", "no", "ok", "okay", "thanks", "thank", "please", "sorry", "wow", "omg",
            "tbh", "imo", "imho", "fwiw", "btw", "afaik", "tldr", "tl", "dr", "etc", "ie", "eg",
            
            # Numbers and common short words (less than 3 characters)
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "30", "am", "pm", "re", "vs", "id", "tv", "pc", "ai"
        }
        
        # Meaningful POS tags to keep (nouns, adjectives, proper nouns, some verbs)
        self.meaningful_pos_tags = {
            'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
            'JJ', 'JJR', 'JJS',          # Adjectives 
            'VBG',                        # Gerunds (often meaningful)
            'VBN',                        # Past participles (often meaningful)
            'RB', 'RBR', 'RBS'           # Adverbs (some are meaningful)
        }

    def _extract_meaningful_words(self, text: str) -> List[str]:
        """Extract meaningful words using NLP techniques."""
        # Clean text
        text = re.sub(r"https?://\S+", "", text)  # Remove URLs
        text = re.sub(r"@\w+", "", text)          # Remove mentions
        text = re.sub(r"#(\w+)", r"\1", text)     # Keep hashtag content
        text = re.sub(r"[^\w\s]", " ", text)      # Remove punctuation
        text = re.sub(r"\d+", "", text)           # Remove standalone numbers
        
        # Tokenize and get POS tags
        tokens = word_tokenize(text.lower())
        pos_tagged = pos_tag(tokens)
        
        meaningful_words = []
        for word, pos in pos_tagged:
            # Filter by length, stopwords, and POS tags
            if (len(word) >= 3 and 
                word not in self.stopwords and 
                pos in self.meaningful_pos_tags):
                meaningful_words.append(word)
        
        return meaningful_words

    def _get_word_frequencies(self, posts: List["Post"], sentiment_filter: str = None) -> Dict[str, int]:
        """Get word frequencies from posts with intelligent filtering."""
        if sentiment_filter:
            posts = [p for p in posts if hasattr(p, "sentiment") and 
                    p.sentiment.get("label") == sentiment_filter]
        
        if not posts:
            return {}
        
        all_meaningful_words = []
        
        for post in posts:
            # Combine title and content
            full_text = f"{post.title or ''} {post.content or ''}"
            meaningful_words = self._extract_meaningful_words(full_text)
            all_meaningful_words.extend(meaningful_words)
        
        # Count word frequencies
        word_freq = Counter(all_meaningful_words)
        
        # Filter out very rare words (appear less than 2 times if we have many posts)
        min_freq = 2 if len(posts) > 20 else 1
        filtered_freq = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
        
        return filtered_freq

    def create_wordcloud(self, posts: List["Post"], output_path: str, sentiment_filter: str = None) -> None:
        """Create an intelligent word cloud from posts."""
        if not posts:
            return

        # Get meaningful word frequencies
        word_frequencies = self._get_word_frequencies(posts, sentiment_filter)
        
        if not word_frequencies:
            # Create a placeholder if no meaningful words found
            word_frequencies = {"no_meaningful_content": 1}

        # Choose colors based on sentiment
        if sentiment_filter == "POSITIVE":
            colormap = "Greens"
        elif sentiment_filter == "NEGATIVE":
            colormap = "Reds"
        else:
            colormap = "viridis"

        # Create word cloud with intelligent word selection
        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            max_words=100,
            relative_scaling=0.5,
            colormap=colormap,
            prefer_horizontal=0.7,
            min_font_size=10,
            max_font_size=100,
            scale=2,
            collocations=False  # Avoid pairing words
        ).generate_from_frequencies(word_frequencies)

        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        
        # Set title
        title = "Intelligent Word Cloud"
        if sentiment_filter:
            title += f" - {sentiment_filter.title()} Sentiment"
        ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
        
        # Add subtitle with word count
        subtitle = f"Showing {len(word_frequencies)} meaningful terms"
        if sentiment_filter:
            filtered_posts = [p for p in posts if hasattr(p, "sentiment") and 
                            p.sentiment.get("label") == sentiment_filter]
            subtitle += f" from {len(filtered_posts)} posts"
        else:
            subtitle += f" from {len(posts)} posts"
        
        plt.figtext(0.5, 0.02, subtitle, ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
