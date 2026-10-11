#!/usr/bin/env python3
"""
Markov Chatbot Studio v3.0
Build chatbots using Markov Chains & Markov Decision Processes.

New in v3.0:
  • Multi-model management (create, rename, switch, delete personalities)
  • Real-time learning from conversations
  • Perplexity score calculation
  • Transition probability heatmap
  • Interactive MDP state explorer
  • Multi-turn conversation context
  • Quick reply suggestions
  • Batch response generation with ranking
  • Theme system (Dark Nebula / Ocean Deep / Emerald Night / Solar Flare)
  • Toast notification system
  • Search in chat history
  • Drag & drop file loading
  • Auto-save timer
  • Markdown & JSON export
  • Sentence completion mode
  • Conversation topic tracking
  • Keyboard shortcut overlay
  • Collapsible panels
  • Status indicators
  • Animated tab badges
  • Better error handling & recovery

Requires: pip install PySide6
"""

import sys, os, random, re, math, pickle, time, json, hashlib
from collections import defaultdict, Counter
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QFormLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QProgressBar,
    QMessageBox, QSplitter, QComboBox, QCheckBox, QFileDialog,
    QStatusBar, QScrollArea, QFrame, QSlider, QDialog,
    QDialogButtonBox, QSizePolicy, QToolTip, QToolButton,
    QMenu, QInputDialog, QListWidget, QListWidgetItem,
    QAbstractItemView, QGridLayout, QProgressBar
)
from PySide6.QtCore import (
    Qt, Signal, QPointF, QRectF, QTimer, QElapsedTimer, QSize,
    QPropertyAnimation, QEasingCurve, QByteArray, QMimeData,
    QPoint, QRect
)
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QRadialGradient,
    QLinearGradient, QFont, QPalette, QAction, QPainterPath,
    QTransform, QWheelEvent, QMouseEvent, QDragEnterEvent,
    QDropEvent, QPixmap, QIcon, QConicalGradient, QFontMetrics
)

# ════════════════════════════════════════════════════════════════
#  CONSTANTS & THEMES
# ════════════════════════════════════════════════════════════════

ACCENT_MC  = "#e94560"
ACCENT_MDP = "#00b4d8"
ACCENT_CMP = "#f0a500"
ACCENT_ANA = "#43aa8b"
ACCENT_MGR = "#a855f7"

STOPWORDS = frozenset(
    "a an the and or but is are was were be been being have has had do does did "
    "will would shall should may might must can could of in on at to for with "
    "by from as into about between through during before after above below "
    "up down out off over under again further then once here there when where "
    "why how all both each few more most other some such no not only own same "
    "so than too very just because it its i me my we our you your he him his "
    "she her they them their this that these those am".split()
)

THEMES = {
    "Dark Nebula": {
        "bg": "#0a0a1a", "bg2": "#16213e", "bg3": "#1a1a2e",
        "border": "#0f3460", "text": "#e0e0e0", "dim": "#8888aa",
        "accent": "#e94560", "accent2": "#533483", "input": "#16213e",
        "chat_bg": "#1a1a2e", "header_grad": "stop:0 #533483, stop:0.5 #e94560, stop:1 #00b4d8",
    },
    "Ocean Deep": {
        "bg": "#0a1628", "bg2": "#0d2137", "bg3": "#0f2b46",
        "border": "#1a4a7a", "text": "#d0e8ff", "dim": "#5a8aaa",
        "accent": "#00b4d8", "accent2": "#0077b6", "input": "#0d2137",
        "chat_bg": "#0f2b46", "header_grad": "stop:0 #0077b6, stop:0.5 #00b4d8, stop:1 #48cae4",
    },
    "Emerald Night": {
        "bg": "#0a1a0f", "bg2": "#0f2a16", "bg3": "#14361c",
        "border": "#1a5a2a", "text": "#c8e8d0", "dim": "#5a9a6a",
        "accent": "#43aa8b", "accent2": "#2d6a4f", "input": "#0f2a16",
        "chat_bg": "#14361c", "header_grad": "stop:0 #2d6a4f, stop:0.5 #43aa8b, stop:1 #90be6d",
    },
    "Solar Flare": {
        "bg": "#1a0f0a", "bg2": "#2a1610", "bg3": "#361e14",
        "border": "#5a2a1a", "text": "#ffe8d0", "dim": "#aa8a5a",
        "accent": "#f0a500", "accent2": "#e76f51", "input": "#2a1610",
        "chat_bg": "#361e14", "header_grad": "stop:0 #e76f51, stop:0.5 #f0a500, stop:1 #f4d35e",
    },
}

CURRENT_THEME = "Dark Nebula"

def theme():
    return THEMES[CURRENT_THEME]

SAMPLE_TEXT = (
    "Hello there! How are you doing today? I am doing great thanks for asking. "
    "The weather is beautiful today. I love sunny days. Sunny days make me happy and energetic. "
    "What is your favorite color? My favorite color is blue. Blue like the sky on a clear sunny day. "
    "Tell me about yourself. I am a chatbot built with markov chains and markov decision processes. "
    "Markov chains predict the next state based on the current state. The current state depends on previous transitions. "
    "Transitions have probabilities that determine how likely each next state is. Each state can transition to multiple possible states. "
    "What do you like to do? I like to chat with people about various topics. Chatting is fun and educational. "
    "How does a markov chain work? A markov chain uses transition probabilities to generate sequences. "
    "The sequences depend on the order of the chain. Higher order chains remember more context. "
    "Context helps produce more coherent output. Coherent output makes for better conversation. "
    "What is the weather like? The weather is sunny and warm today. Warm weather is pleasant and enjoyable. "
    "Pleasant weather makes people happy and sociable. Happy people are friendly and talkative. "
    "I enjoy learning about artificial intelligence. AI is a fascinating and rapidly growing field. "
    "Machine learning is a powerful subset of artificial intelligence. Deep learning uses neural networks with many layers. "
    "Neural networks are inspired by the human brain. The brain processes information through interconnected neurons. "
    "Markov decision processes add rewards and policies to markov chains. Rewards measure how good an action is in a state. "
    "Policies determine the best action to take. Value iteration computes optimal values for each state. "
    "The discount factor gamma controls how much future rewards matter. A high gamma means the agent values long term outcomes. "
    "Q-learning is a model free reinforcement learning algorithm. It learns optimal policies through exploration and exploitation. "
    "What is your name? I am a markov chatbot. You can call me Markov. Nice to meet you Markov. "
    "Tell me a joke. Why did the markov chain break up? Because it had no memory of the past. "
    "That is funny. I am glad you liked it. Do you know any other jokes? Why was the MDP sad? "
    "Because it could not find an optimal policy. Optimization can be difficult but rewarding. "
    "What can you do? I can generate text using markov chains. I can also choose optimal responses using MDP. "
    "Both approaches have strengths and weaknesses. Markov chains are simple and fast. MDP is more intelligent but slower. "
    "Can you learn from new data? Yes you can train me with new text. The more data the better the responses. "
    "Good data leads to good conversation. Quality matters more than quantity sometimes. "
    "Thank you for chatting with me. You are welcome. I enjoyed our conversation. Let us talk again soon. "
    "Programming is both an art and a science. Good code is clean readable and maintainable. "
    "Python is a versatile and popular programming language. Data science uses python for analysis and visualization. "
    "Reinforcement learning trains agents through rewards and penalties. The agent explores the environment to maximize cumulative reward. "
    "Natural language processing helps computers understand human language. Chatbots use NLP to converse with users naturally."
    "The quick brown fox jumps over the lazy dog. Dogs are loyal and friendly animals. Cats are independent and graceful creatures. "
    "Music brings joy and peace to the soul. Art expresses what words cannot describe. "
    "Science explores the mysteries of the universe. Technology transforms how we live and work. "
    "Education opens doors to new opportunities. Knowledge is power and wisdom is its companion. "
    "Travel broadens the mind and enriches the spirit. Adventure awaits those who seek it."
)


# ════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ════════════════════════════════════════════════════════════════

class PreprocessConfig:
    def __init__(self, lowercase=True, remove_punct=False,
                 remove_nums=False, remove_stops=False):
        self.lowercase    = lowercase
        self.remove_punct = remove_punct
        self.remove_nums  = remove_nums
        self.remove_stops = remove_stops

    def apply(self, text):
        if self.lowercase:
            text = text.lower()
        if self.remove_punct:
            text = re.sub(r'[^\w\s]', ' ', text)
        if self.remove_nums:
            text = re.sub(r'\b\d+\b', ' ', text)
        if self.remove_stops:
            text = ' '.join(w for w in text.split() if w not in STOPWORDS)
        return re.sub(r'\s+', ' ', text).strip()


# ════════════════════════════════════════════════════════════════
#  TOAST NOTIFICATION SYSTEM
# ════════════════════════════════════════════════════════════════

class ToastManager(QWidget):
    """Floating toast notifications that stack in the top-right corner."""
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._toasts = []
        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignTop | Qt.AlignRight)
        self._layout.setSpacing(6)
        self._layout.setContentsMargins(0, 0, 10, 0)
        self.resize(400, 600)

    @staticmethod
    def show(message, kind="info", duration=3000):
        mgr = ToastManager.instance()
        mgr._show(message, kind, duration)

    def _show(self, message, kind, duration):
        colors = {
            "info":    ("#16213e", "#00b4d8", "ℹ️"),
            "success": ("#0f2a16", "#43aa8b", "✅"),
            "warning": ("#2a1610", "#f0a500", "⚠️"),
            "error":   ("#2a0f0f", "#e94560", "❌"),
        }
        bg, accent, icon = colors.get(kind, colors["info"])
        toast = QFrame()
        toast.setStyleSheet(f"""
            QFrame {{ background:{bg}; border:2px solid {accent};
                      border-radius:10px; padding:8px 14px; }}
            QLabel {{ color:{accent}; font-size:13px; border:none; }}
        """)
        tl = QHBoxLayout(toast)
        tl.setContentsMargins(6, 4, 6, 4)
        icon_lbl = QLabel(icon)
        icon_lbl.setFixedSize(22, 22)
        msg_lbl = QLabel(message)
        msg_lbl.setWordWrap(True)
        msg_lbl.setMaximumWidth(300)
        tl.addWidget(icon_lbl)
        tl.addWidget(msg_lbl, 1)
        self._layout.addWidget(toast)
        self._toasts.append(toast)

        # Position over parent window
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QMainWindow):
                g = w.geometry()
                self.move(g.right() - self.width() - 10, g.top() + 50)
                break
        self.show()
        self.adjustSize()

        QTimer.singleShot(duration, lambda: self._remove(toast))

    def _remove(self, toast):
        if toast in self._toasts:
            self._toasts.remove(toast)
            self._layout.removeWidget(toast)
            toast.deleteLater()
        if not self._toasts:
            self.hide()


# ════════════════════════════════════════════════════════════════
#  MARKOV CHAIN ENGINE
# ════════════════════════════════════════════════════════════════

class MarkovChain:
    def __init__(self, order=2, preprocess=None):
        self.order = order
        self.preprocess = preprocess or PreprocessConfig()
        self.transitions = defaultdict(Counter)
        self.starts = []
        self.vocabulary = set()
        self.trained = False
        self._raw = ""
        self._context_history = []  # multi-turn context

    def train(self, text):
        self._raw = text
        proc = self.preprocess.apply(text)
        sentences = re.split(r'[.!?;\n]+', proc)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue
            self.vocabulary.update(words)
            padded = [None] * self.order + words + [None]
            self.starts.append(tuple(padded[:self.order]))
            for i in range(len(padded) - self.order):
                state = tuple(padded[i:i + self.order])
                nxt = padded[i + self.order]
                self.transitions[state][nxt] += 1
        self.trained = True

    def learn_from_conversation(self, text):
        """Incrementally learn from new text without resetting."""
        if not text.strip():
            return
        self.train(text)

    @staticmethod
    def _pick(counter, temperature=1.0):
        items = list(counter.items())
        if not items:
            return None
        t = max(temperature, 0.05)
        weights = [c ** (1.0 / t) for _, c in items]
        total = sum(weights)
        r = random.uniform(0, total)
        cum = 0
        for (word, _), w in zip(items, weights):
            cum += w
            if r <= cum:
                return word
        return items[-1][0]

    def generate(self, seed=None, max_words=50, temperature=1.0,
                 context=None, completion_mode=False):
        if not self.trained:
            return "⚠ Model not trained yet.", 0.0

        # Build state from context + seed
        context_words = []
        if context:
            for msg in context[-3:]:  # last 3 messages
                context_words.extend(self.preprocess.apply(msg).split())
            context_words = context_words[-(self.order * 2):]  # limit

        if seed:
            sw = self.preprocess.apply(seed).split()
            all_words = context_words + sw
        else:
            all_words = context_words

        if len(all_words) >= self.order:
            state = tuple(all_words[-self.order:])
        elif all_words:
            state = tuple([None] * (self.order - len(all_words)) + all_words)
        else:
            state = random.choice(self.starts) if self.starts else tuple([None] * self.order)

        # Fallback if state not in transitions
        if state not in self.transitions:
            state = random.choice(self.starts) if self.starts else tuple([None] * self.order)

        result = [w for w in state if w is not None]
        probs = []

        # In completion mode, prepend seed words
        if completion_mode and seed:
            seed_display = seed.split()
        else:
            seed_display = []

        for _ in range(max_words):
            if state not in self.transitions:
                break
            counter = self.transitions[state]
            total = sum(counter.values())
            nxt = self._pick(counter, temperature)
            if nxt is None:
                break
            probs.append(counter[nxt] / total if total else 0)
            result.append(nxt)
            state = tuple(list(state[1:]) + [nxt])

        if not result:
            return "🤔 Couldn't generate anything.", 0.0

        text = self._post_process(result)
        confidence = sum(probs) / len(probs) if probs else 0.0
        return text, confidence

    def generate_batch(self, seed=None, max_words=50, temperature=1.0,
                       context=None, count=3):
        """Generate multiple responses and rank by confidence."""
        results = []
        for _ in range(count):
            text, conf = self.generate(seed=seed, max_words=max_words,
                                        temperature=temperature, context=context)
            if not text.startswith("⚠") and not text.startswith("🤔"):
                results.append((text, conf))
        results.sort(key=lambda x: x[1], reverse=True)
        return results if results else [("No valid responses generated.", 0.0)]

    @staticmethod
    def _post_process(words):
        text = ' '.join(words)
        if text:
            text = text[0].upper() + text[1:]
        if text and text[-1] not in '.!?':
            text += '.'
        parts = re.split(r'([.!?]\s*)', text)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts) and parts[i + 1]:
                parts[i + 1] = parts[i + 1][0].upper() + parts[i + 1][1:]
        return ''.join(parts)

    def get_probability(self, state, word):
        if state not in self.transitions:
            return 0.0
        total = sum(self.transitions[state].values())
        return self.transitions[state].get(word, 0) / total if total else 0.0

    def compute_perplexity(self, text):
        """Compute perplexity of given text under the model."""
        if not self.trained:
            return float('inf')
        proc = self.preprocess.apply(text)
        words = proc.split()
        if len(words) < self.order + 1:
            return float('inf')
        padded = [None] * self.order + words
        log_prob = 0.0
        n = 0
        for i in range(len(padded) - self.order):
            state = tuple(padded[i:i + self.order])
            nxt = padded[i + self.order] if i + self.order < len(padded) else None
            if nxt is None:
                continue
            prob = self.get_probability(state, nxt)
            if prob > 0:
                log_prob += math.log2(prob)
            else:
                log_prob += math.log2(1e-10)
            n += 1
        if n == 0:
            return float('inf')
        cross_entropy = -log_prob / n
        return 2 ** cross_entropy

    def suggest_replies(self, context=None, count=3):
        """Suggest what the user might say next."""
        if not self.trained:
            return []
        suggestions = []
        for _ in range(count * 3):  # generate more, pick diverse
            text, conf = self.generate(max_words=12, temperature=1.5,
                                        context=context)
            if not text.startswith("⚠") and not text.startswith("🤔"):
                suggestions.append(text)
        # Deduplicate and return top count
        seen = set()
        unique = []
        for s in suggestions:
            key = s.lower()[:30]
            if key not in seen:
                seen.add(key)
                unique.append(s)
            if len(unique) >= count:
                break
        return unique

    def get_graph_data(self, max_nodes=40):
        nodes, edges = set(), []
        for state, trans in self.transitions.items():
            label = " ".join(w for w in state if w)
            if label:
                nodes.add(label)
            for nxt, count in trans.items():
                if nxt is not None:
                    nodes.add(nxt)
                    total = sum(trans.values())
                    prob = count / total if total else 0
                    edges.append((label, nxt, prob))
        if len(nodes) > max_nodes:
            nodes = set(list(nodes)[:max_nodes])
            edges = [(s, t, p) for s, t, p in edges if s in nodes and t in nodes]
        return nodes, edges

    def get_word_frequencies(self, top_n=25):
        freq = Counter()
        for counter in self.transitions.values():
            for word, count in counter.items():
                if word is not None:
                    freq[word] += count
        return freq.most_common(top_n)

    def get_top_transitions(self, top_n=20):
        all_trans = []
        for state, trans in self.transitions.items():
            label = " ".join(w for w in state if w)
            for nxt, count in trans.items():
                if nxt is not None:
                    total = sum(trans.values())
                    prob = count / total if total else 0
                    all_trans.append((label, nxt, count, prob))
        all_trans.sort(key=lambda x: x[2], reverse=True)
        return all_trans[:top_n]

    def get_transition_matrix(self, top_n=15):
        """Get top-N states and their transition probability matrix."""
        state_counts = Counter()
        for state, trans in self.transitions.items():
            label = " ".join(w for w in state if w)
            state_counts[label] = sum(trans.values())
        top_states = [s for s, _ in state_counts.most_common(top_n)]
        state_set = set(top_states)
        matrix = {}
        for state, trans in self.transitions.items():
            label = " ".join(w for w in state if w)
            if label not in state_set:
                continue
            total = sum(trans.values())
            for nxt, count in trans.items():
                if nxt is not None and nxt in state_set:
                    matrix[(label, nxt)] = count / total if total else 0
        return top_states, matrix

    def get_stats(self):
        return {
            "order": self.order,
            "states": len(self.transitions),
            "transitions": sum(len(v) for v in self.transitions.values()),
            "vocabulary": len(self.vocabulary),
            "raw_chars": len(self._raw),
        }


# ════════════════════════════════════════════════════════════════
#  MDP ENGINE
# ════════════════════════════════════════════════════════════════

class MDPChatbot:
    def __init__(self, gamma=0.9, theta=0.001, preprocess=None):
        self.gamma = gamma
        self.theta = theta
        self.preprocess = preprocess or PreprocessConfig()
        self.states      = {}
        self.transitions = defaultdict(dict)
        self.rewards     = defaultdict(dict)
        self.values      = {}
        self.policy      = {}
        self.q_table     = defaultdict(dict)
        self.trained     = False
        self._word_index = defaultdict(list)
        self.convergence_history = []
        self._feedback_log = []
        self._last_response_sid = None

    def train(self, text):
        raw = re.split(r'[.!?;\n]+', text)
        sentences = [self.preprocess.apply(s) for s in raw if len(s.strip()) > 3]
        sentences = [s for s in sentences if s]
        if len(sentences) < 2:
            return

        for i, s in enumerate(sentences):
            words = set(s.split())
            self.states[i] = {"text": s, "words": words}
            for w in words:
                self._word_index[w].append(i)

        for sid, info in self.states.items():
            candidates = []
            if sid + 1 in self.states:
                candidates.append((sid + 1, 0.35))
            related = Counter()
            for w in info["words"]:
                for other in self._word_index[w]:
                    if other != sid:
                        related[other] += 1
            for other, cnt in related.most_common(5):
                candidates.append((other, 0.08 * cnt))
            others = [s for s in self.states if s != sid]
            if others:
                for r in random.sample(others, min(3, len(others))):
                    candidates.append((r, 0.03))
            total_p = sum(p for _, p in candidates)
            if total_p > 0:
                candidates = [(s, p / total_p) for s, p in candidates]
            for aid, (nsid, prob) in enumerate(candidates):
                self.transitions[sid][aid] = [(nsid, prob)]
                cur_w = info["words"]
                nxt_w = self.states[nsid]["words"]
                inter = len(cur_w & nxt_w)
                union = len(cur_w | nxt_w) or 1
                reward = inter / union
                if nsid == sid + 1:
                    reward += 0.5
                self.rewards[sid][aid] = reward

        for sid in self.states:
            self.values[sid] = 0.0
        self.trained = True
        self.convergence_history = []

    def learn_from_conversation(self, text):
        """Incrementally add new sentences as states."""
        if not text.strip():
            return
        new_sentences = [self.preprocess.apply(s) for s in re.split(r'[.!?;\n]+', text) if len(s.strip()) > 3]
        start_id = len(self.states)
        for i, s in enumerate(new_sentences):
            sid = start_id + i
            words = set(s.split())
            self.states[sid] = {"text": s, "words": words}
            self.values[sid] = 0.0
            for w in words:
                self._word_index[w].append(sid)

        # Re-link transitions for new states
        for sid in range(start_id, len(self.states)):
            info = self.states[sid]
            candidates = []
            if sid + 1 in self.states:
                candidates.append((sid + 1, 0.35))
            related = Counter()
            for w in info["words"]:
                for other in self._word_index[w]:
                    if other != sid:
                        related[other] += 1
            for other, cnt in related.most_common(5):
                candidates.append((other, 0.08 * cnt))
            others = [s for s in self.states if s != sid]
            if others:
                for r in random.sample(others, min(2, len(others))):
                    candidates.append((r, 0.03))
            total_p = sum(p for _, p in candidates)
            if total_p > 0:
                candidates = [(s, p / total_p) for s, p in candidates]
            for aid, (nsid, prob) in enumerate(candidates):
                self.transitions[sid][aid] = [(nsid, prob)]
                cur_w = info["words"]
                nxt_w = self.states[nsid]["words"]
                inter = len(cur_w & nxt_w)
                union = len(cur_w | nxt_w) or 1
                reward = inter / union
                if nsid == sid + 1:
                    reward += 0.5
                self.rewards[sid][aid] = reward

    def value_iteration_step(self):
        delta = 0.0
        for sid in self.states:
            if sid not in self.transitions:
                continue
            qs = []
            for aid in self.transitions[sid]:
                q = self.rewards[sid].get(aid, 0)
                for nsid, prob in self.transitions[sid][aid]:
                    q += self.gamma * prob * self.values.get(nsid, 0)
                qs.append(q)
            if qs:
                new_v = max(qs)
                delta = max(delta, abs(new_v - self.values[sid]))
                self.values[sid] = new_v
        return delta

    def run_value_iteration(self, max_iter=200, theta=0.001, callback=None):
        self.convergence_history = []
        for i in range(max_iter):
            delta = self.value_iteration_step()
            self.convergence_history.append((i + 1, delta))
            if callback:
                callback(i + 1, delta)
            if delta < theta:
                break
        self.extract_policy()

    def run_q_learning(self, episodes=500, alpha=0.1, epsilon=0.2,
                       max_steps=20, callback=None):
        self.convergence_history = []
        all_sids = list(self.states.keys())
        if not all_sids:
            return
        for ep in range(episodes):
            sid = random.choice(all_sids)
            total_delta = 0.0
            for step in range(max_steps):
                if sid not in self.transitions:
                    break
                actions = list(self.transitions[sid].keys())
                if not actions:
                    break
                if random.random() < epsilon:
                    aid = random.choice(actions)
                else:
                    aid = max(actions, key=lambda a: self.q_table[sid].get(a, 0))
                trans = self.transitions[sid][aid]
                nsid = random.choices([s for s, _ in trans],
                                      weights=[p for _, p in trans])[0]
                reward = self.rewards[sid].get(aid, 0)
                future_actions = list(self.transitions.get(nsid, {}).keys())
                max_future = max((self.q_table[nsid].get(a, 0) for a in future_actions),
                                 default=0)
                old_q = self.q_table[sid].get(aid, 0)
                new_q = old_q + alpha * (reward + self.gamma * max_future - old_q)
                total_delta += abs(new_q - old_q)
                self.q_table[sid][aid] = new_q
                sid = nsid
            self.convergence_history.append((ep + 1, total_delta))
            if callback:
                callback(ep + 1, total_delta)
        for sid in self.states:
            if sid in self.q_table and self.q_table[sid]:
                best_aid = max(self.q_table[sid], key=self.q_table[sid].get)
                self.values[sid] = self.q_table[sid][best_aid]
                if best_aid in self.transitions.get(sid, {}):
                    trans = self.transitions[sid][best_aid]
                    self.policy[sid] = max(trans, key=lambda x: x[1])[0]

    def extract_policy(self):
        for sid in self.states:
            if sid not in self.transitions:
                continue
            best_aid, best_q = None, float("-inf")
            for aid in self.transitions[sid]:
                q = self.rewards[sid].get(aid, 0)
                for nsid, prob in self.transitions[sid][aid]:
                    q += self.gamma * prob * self.values.get(nsid, 0)
                if q > best_q:
                    best_q = q
                    best_aid = aid
            if best_aid is not None and self.transitions[sid][best_aid]:
                self.policy[sid] = max(
                    self.transitions[sid][best_aid], key=lambda x: x[1])[0]

    def respond(self, user_input, context=None):
        if not self.trained:
            return "⚠ Please train the model first.", 0.0

        words = set(self.preprocess.apply(user_input).split())
        # Also consider context
        context_words = set()
        if context:
            for msg in context[-2:]:
                context_words.update(self.preprocess.apply(msg).split())
        all_words = words | context_words

        if not all_words:
            sid = random.choice(list(self.states.keys()))
        else:
            scores = {}
            for sid, info in self.states.items():
                overlap = len(all_words & info["words"])
                if overlap:
                    scores[sid] = overlap + self.values.get(sid, 0) * 0.3
            sid = max(scores, key=scores.get) if scores else random.choice(list(self.states.keys()))

        visited = set()
        current = best = sid
        for _ in range(3):
            if current in self.policy and self.policy[current] not in visited:
                visited.add(current)
                current = self.policy[current]
            else:
                break

        result_sid = current if current in self.states else best
        self._last_response_sid = result_sid
        text = self.states[result_sid]["text"]
        if text:
            text = text[0].upper() + text[1:]
            if text[-1] not in '.!?':
                text += '.'
        confidence = min(1.0, self.values.get(sid, 0) / max(self.values.values() or [1]))
        return text, confidence

    def apply_feedback(self, user_input, response_text, positive=True):
        best_sid, best_aid = None, None
        best_overlap = 0
        for sid, info in self.states.items():
            for aid in self.transitions.get(sid, {}):
                for nsid, _ in self.transitions[sid][aid]:
                    if nsid in self.states and self.states[nsid]["text"][:20] in response_text:
                        overlap = len(info["words"])
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_sid = sid
                            best_aid = aid
        if best_sid is not None and best_aid is not None:
            delta = 0.2 if positive else -0.3
            self.rewards[best_sid][best_aid] = max(
                0, self.rewards[best_sid].get(best_aid, 0) + delta)
            self._feedback_log.append((best_sid, best_aid, delta))
            return True
        return False

    def get_graph_data(self, max_nodes=30):
        nodes, edges = set(), []
        for sid in list(self.states.keys())[:max_nodes]:
            nodes.add(sid)
            if sid in self.policy and self.policy[sid] in self.states:
                tgt = self.policy[sid]
                nodes.add(tgt)
                edges.append((sid, tgt, self.values.get(sid, 0)))
        return nodes, edges

    def get_stats(self):
        vals = list(self.values.values()) if self.values else [0]
        return {
            "states": len(self.states),
            "actions": sum(len(a) for a in self.transitions.values()),
            "avg_value": sum(vals) / len(vals),
            "max_value": max(vals),
            "min_value": min(vals),
            "gamma": self.gamma,
            "feedback_count": len(self._feedback_log),
        }


# ════════════════════════════════════════════════════════════════
#  CONVERGENCE CHART
# ════════════════════════════════════════════════════════════════

class ConvergenceChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = []
        self.setMinimumSize(300, 150)
        self._log_scale = True

    def set_data(self, data):
        self.data = data
        self.update()

    def clear(self):
        self.data = []
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        margin_l, margin_b = 55, 25
        t = theme()

        bg = QLinearGradient(0, 0, 0, H)
        bg.setColorAt(0, QColor(t["bg3"]))
        bg.setColorAt(1, QColor(t["bg"]))
        p.fillRect(self.rect(), bg)

        if len(self.data) < 2:
            p.setPen(QColor(t["dim"]))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(self.rect(), Qt.AlignCenter, "Run solver to see convergence")
            p.end()
            return

        iters = [d[0] for d in self.data]
        deltas = [max(d[1], 1e-15) for d in self.data]
        max_iter = max(iters)
        log_min = math.log10(min(deltas))
        log_max = math.log10(max(deltas))
        if log_max == log_min:
            log_max = log_min + 1
        chart_x = margin_l
        chart_y = 15
        chart_w = W - margin_l - 15
        chart_h = H - margin_b - 15

        # Grid
        p.setPen(QPen(QColor(t["border"]), 0.5, Qt.DotLine))
        for i in range(5):
            y = chart_y + chart_h * i / 4
            p.drawLine(chart_x, int(y), chart_x + chart_w, int(y))
        p.setPen(QColor(t["dim"]))
        p.setFont(QFont("Consolas", 7))
        for i in range(5):
            frac = 1.0 - i / 4.0
            val = 10 ** (log_min + frac * (log_max - log_min))
            y = chart_y + chart_h * i / 4
            p.drawText(0, int(y) - 5, margin_l - 4, 12,
                       Qt.AlignRight | Qt.AlignVCenter, f"{val:.1e}")

        # Plot
        path = QPainterPath()
        for idx, (it, delta) in enumerate(self.data):
            x = chart_x + (it / max_iter) * chart_w
            log_d = math.log10(max(delta, 1e-15))
            y_frac = (log_d - log_min) / (log_max - log_min)
            y = chart_y + chart_h * (1.0 - y_frac)
            if idx == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        fill_path = QPainterPath(path)
        fill_path.lineTo(chart_x + (iters[-1] / max_iter) * chart_w, chart_y + chart_h)
        fill_path.lineTo(chart_x, chart_y + chart_h)
        fill_path.closeSubpath()
        grad = QLinearGradient(0, chart_y, 0, chart_y + chart_h)
        accent = t["accent"]
        grad.setColorAt(0, QColor(accent + "60") if len(accent) == 7 else QColor(accent))
        grad.setColorAt(1, QColor(accent + "05") if len(accent) == 7 else QColor(accent))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawPath(fill_path)

        p.setPen(QPen(QColor(accent), 2))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        p.end()


# ════════════════════════════════════════════════════════════════
#  WORD FREQUENCY CHART
# ════════════════════════════════════════════════════════════════

class WordFreqChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = []
        self.setMinimumSize(300, 200)

    def set_data(self, data):
        self.data = data[:20]
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        t = theme()

        bg = QLinearGradient(0, 0, 0, H)
        bg.setColorAt(0, QColor(t["bg3"]))
        bg.setColorAt(1, QColor(t["bg"]))
        p.fillRect(self.rect(), bg)

        if not self.data:
            p.setPen(QColor(t["dim"]))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(self.rect(), Qt.AlignCenter, "No data")
            p.end()
            return

        n = len(self.data)
        max_count = max(c for _, c in self.data)
        bar_h = max(12, min(20, (H - 20) // n - 2))
        left_margin = 80
        bar_max_w = W - left_margin - 50

        for i, (word, count) in enumerate(self.data):
            y = 10 + i * (bar_h + 3)
            if y + bar_h > H:
                break
            p.setPen(QColor(t["text"]))
            p.setFont(QFont("Consolas", 8))
            p.drawText(0, y, left_margin - 5, bar_h,
                       Qt.AlignRight | Qt.AlignVCenter, word[:10])
            w = (count / max_count) * bar_max_w if max_count else 0
            hue = (i * 19 + 160) % 360
            color = QColor.fromHsv(hue, 180, 210)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            p.drawRoundedRect(QRectF(left_margin, y, w, bar_h), 3, 3)
            p.setPen(Qt.white)
            p.setFont(QFont("Consolas", 8))
            p.drawText(int(left_margin + w + 5), y, 40, bar_h,
                       Qt.AlignLeft | Qt.AlignVCenter, str(count))
        p.end()


# ════════════════════════════════════════════════════════════════
#  HEATMAP WIDGET
# ════════════════════════════════════════════════════════════════

class HeatmapWidget(QWidget):
    """Transition probability heatmap."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.labels = []
        self.matrix = {}
        self.setMinimumSize(300, 300)

    def set_data(self, labels, matrix):
        self.labels = labels
        self.matrix = matrix
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        t = theme()

        p.fillRect(self.rect(), QColor(t["bg"]))

        if not self.labels:
            p.setPen(QColor(t["dim"]))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(self.rect(), Qt.AlignCenter, "Train a model to see heatmap")
            p.end()
            return

        n = len(self.labels)
        margin_left = 70
        margin_top = 50
        cell_w = min(28, (W - margin_left - 10) / n)
        cell_h = min(22, (H - margin_top - 10) / n)
        grid_w = cell_w * n
        grid_h = cell_h * n

        # Title
        p.setPen(QColor(t["text"]))
        p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        p.drawText(QRect(0, 5, W, 20), Qt.AlignCenter, "Transition Probability Heatmap")

        # Column labels
        p.setFont(QFont("Consolas", 6))
        p.save()
        for j, lbl in enumerate(self.labels):
            x = margin_left + j * cell_w + cell_w / 2
            p.translate(x, margin_top - 4)
            p.rotate(-45)
            p.drawText(QRect(-40, 0, 80, 12), Qt.AlignCenter, lbl[:8])
            p.resetTransform()
        p.restore()

        # Row labels
        for i, lbl in enumerate(self.labels):
            y = margin_top + i * cell_h + cell_h / 2
            p.setPen(QColor(t["text"]))
            p.setFont(QFont("Consolas", 6))
            p.drawText(QRect(0, int(y - 6), margin_left - 4, 12),
                       Qt.AlignRight | Qt.AlignVCenter, lbl[:10])

        # Cells
        max_val = max(self.matrix.values()) if self.matrix else 1
        for i, src in enumerate(self.labels):
            for j, tgt in enumerate(self.labels):
                val = self.matrix.get((src, tgt), 0)
                intensity = val / max_val if max_val > 0 else 0
                # Blue to red gradient
                r = int(220 * intensity)
                b = int(220 * (1 - intensity))
                g = int(80 * intensity * (1 - intensity) * 4)
                color = QColor(r, g, b, 200)
                x = margin_left + j * cell_w
                y = margin_top + i * cell_h
                p.setPen(QPen(QColor(t["border"]), 0.5))
                p.setBrush(QBrush(color))
                p.drawRect(QRectF(x, y, cell_w, cell_h))
                # Value text
                if cell_w > 18 and cell_h > 14 and val > 0:
                    p.setPen(Qt.white if intensity > 0.5 else QColor(t["text"]))
                    p.setFont(QFont("Consolas", 6))
                    p.drawText(QRectF(x, y, cell_w, cell_h),
                               Qt.AlignCenter, f"{val:.2f}")

        # Color bar legend
        bar_x = margin_left + grid_w + 10
        bar_y = margin_top
        bar_w = 12
        bar_h = grid_h
        if bar_x + bar_w + 30 < W:
            for y_i in range(int(bar_h)):
                frac = 1.0 - y_i / bar_h
                r = int(220 * frac)
                b = int(220 * (1 - frac))
                g_c = int(80 * frac * (1 - frac) * 4)
                p.setPen(QColor(r, g_c, b))
                p.drawLine(int(bar_x), int(bar_y + y_i),
                           int(bar_x + bar_w), int(bar_y + y_i))
            p.setPen(QColor(t["dim"]))
            p.setFont(QFont("Consolas", 7))
            p.drawText(int(bar_x + bar_w + 2), int(bar_y + 8), f"{max_val:.2f}")
            p.drawText(int(bar_x + bar_w + 2), int(bar_y + bar_h), "0.00")

        p.end()


# ════════════════════════════════════════════════════════════════
#  GRAPH WIDGET (force layout, zoom/pan)
# ════════════════════════════════════════════════════════════════

class GraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nodes = set()
        self.edges = []
        self.positions = {}
        self.velocities = {}
        self.colors = {}
        self._dragging = None
        self._offset = QPointF()
        self._pan_offset = QPointF(0, 0)
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()
        self._hover_node = None
        self._force_timer = QTimer(self)
        self._force_timer.setInterval(30)
        self._force_timer.timeout.connect(self._force_step)
        self._force_temp = 1.0
        self.setMinimumSize(350, 300)
        self.setMouseTracking(True)
        self.setAcceptDrops(False)

    def set_data(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.positions.clear()
        self.velocities.clear()
        self.colors.clear()
        cx, cy = self.width() / 2, self.height() / 2
        r = min(cx, cy) * 0.6
        nlist = list(nodes)
        for i, nd in enumerate(nlist):
            a = 2 * math.pi * i / max(len(nlist), 1) - math.pi / 2
            self.positions[nd] = QPointF(cx + r * math.cos(a), cy + r * math.sin(a))
            self.velocities[nd] = QPointF(0, 0)
            self.colors[nd] = QColor.fromHsv((i * 137 + 200) % 360, 190, 220)
        self._force_temp = 1.0
        self.update()

    def start_force_layout(self):
        self._force_temp = 1.0
        self._force_timer.start()

    def stop_force_layout(self):
        self._force_timer.stop()

    def _force_step(self):
        if not self.positions:
            return
        W, H = self.width(), self.height()
        nodes_list = list(self.positions.keys())
        forces = {n: QPointF(0, 0) for n in nodes_list}

        for i, u in enumerate(nodes_list):
            for j in range(i + 1, len(nodes_list)):
                v = nodes_list[j]
                diff = self.positions[u] - self.positions[v]
                dist = max(math.sqrt(diff.x()**2 + diff.y()**2), 1)
                rep = 4000.0 / (dist * dist)
                fx = (diff.x() / dist) * rep
                fy = (diff.y() / dist) * rep
                forces[u] += QPointF(fx, fy)
                forces[v] -= QPointF(fx, fy)

        for src, tgt, w in self.edges:
            if src not in self.positions or tgt not in self.positions:
                continue
            diff = self.positions[tgt] - self.positions[src]
            dist = max(math.sqrt(diff.x()**2 + diff.y()**2), 1)
            att = dist * 0.01 * max(w, 0.05)
            fx = (diff.x() / dist) * att
            fy = (diff.y() / dist) * att
            forces[src] += QPointF(fx, fy)
            forces[tgt] -= QPointF(fx, fy)

        cx, cy = W / 2, H / 2
        for n in nodes_list:
            diff = QPointF(cx, cy) - self.positions[n]
            forces[n] += diff * 0.002

        damping = 0.85
        for n in nodes_list:
            self.velocities[n] = (self.velocities[n] + forces[n]) * damping * self._force_temp
            self.positions[n] += self.velocities[n]
            self.positions[n] = QPointF(
                max(30, min(W - 30, self.positions[n].x())),
                max(30, min(H - 30, self.positions[n].y())))

        self._force_temp *= 0.97
        if self._force_temp < 0.01:
            self.stop_force_layout()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        t = theme()

        bg = QLinearGradient(0, 0, 0, H)
        bg.setColorAt(0, QColor(t["bg3"]))
        bg.setColorAt(1, QColor(t["bg"]))
        p.fillRect(self.rect(), bg)

        if not self.nodes:
            p.setPen(QColor(t["dim"]))
            p.setFont(QFont("Segoe UI", 11))
            p.drawText(self.rect(), Qt.AlignCenter, "No graph data — train a model first.")
            p.end()
            return

        p.translate(self._pan_offset)
        p.scale(self._zoom, self._zoom)

        for src, tgt, w in self.edges:
            if src not in self.positions or tgt not in self.positions:
                continue
            p1, p2 = self.positions[src], self.positions[tgt]
            intensity = min(255, int(abs(w) * 200 + 55))
            pen = QPen(QColor(80, intensity, 255, 160), max(1.0, abs(w) * 2.0))
            p.setPen(pen)
            p.drawLine(p1, p2)
            ang = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
            sz = 8
            a1 = p2 - QPointF(sz * math.cos(ang - 0.4), sz * math.sin(ang - 0.4))
            a2 = p2 - QPointF(sz * math.cos(ang + 0.4), sz * math.sin(ang + 0.4))
            path = QPainterPath()
            path.moveTo(p2)
            path.lineTo(a1)
            path.lineTo(a2)
            path.closeSubpath()
            p.setBrush(QColor(80, intensity, 255, 160))
            p.setPen(Qt.NoPen)
            p.drawPath(path)
            if abs(w) > 0.01:
                mid = (p1 + p2) / 2
                p.setPen(QColor(180, 180, 220, 180))
                p.setFont(QFont("Consolas", 7))
                p.drawText(mid + QPointF(2, -4), f"{abs(w):.2f}")

        for nd, pos in self.positions.items():
            col = self.colors.get(nd, QColor(120, 160, 255))
            is_hover = (nd == self._hover_node)
            radius = 12 if is_hover else 9
            glow = QRadialGradient(pos, radius * 3)
            glow.setColorAt(0, QColor(col.red(), col.green(), col.blue(),
                                       100 if is_hover else 60))
            glow.setColorAt(1, QColor(col.red(), col.green(), col.blue(), 0))
            p.setBrush(QBrush(glow))
            p.setPen(Qt.NoPen)
            p.drawEllipse(pos, radius * 3, radius * 3)
            p.setBrush(QBrush(col))
            p.setPen(QPen(Qt.white if is_hover else QColor(200, 200, 220),
                          1.5 if is_hover else 0.8))
            p.drawEllipse(pos, radius, radius)
            label = str(nd)[:18]
            p.setPen(Qt.white)
            p.setFont(QFont("Segoe UI", 7 + (1 if is_hover else 0),
                             QFont.Bold if is_hover else QFont.Normal))
            p.drawText(QRectF(pos.x() - 55, pos.y() + radius + 2, 110, 16),
                       Qt.AlignCenter, label)
        p.end()

    def _screen_to_scene(self, pos):
        return (pos - self._pan_offset) / self._zoom

    def _find_node(self, pos):
        sp = self._screen_to_scene(pos)
        for nd, pp in self.positions.items():
            if (sp - pp).manhattanLength() < 16:
                return nd
        return None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            nd = self._find_node(ev.position())
            if nd is not None:
                self._dragging = nd
                self._offset = self._screen_to_scene(ev.position()) - self.positions[nd]
                return
        elif ev.button() in (Qt.RightButton, Qt.MiddleButton):
            self._panning = True
            self._pan_start = ev.position()

    def mouseMoveEvent(self, ev):
        if self._dragging is not None:
            self.positions[self._dragging] = self._screen_to_scene(ev.position()) - self._offset
            self.update()
        elif self._panning:
            self._pan_offset += ev.position() - self._pan_start
            self._pan_start = ev.position()
            self.update()
        else:
            nd = self._find_node(ev.position())
            if nd != self._hover_node:
                self._hover_node = nd
                self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._dragging = None
        elif ev.button() in (Qt.RightButton, Qt.MiddleButton):
            self._panning = False

    def mouseDoubleClickEvent(self, ev):
        self.start_force_layout()

    def wheelEvent(self, event):
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self._zoom = max(0.2, min(5.0, self._zoom * factor))
        self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.nodes:
            self.set_data(self.nodes, self.edges)


# ════════════════════════════════════════════════════════════════
#  CHAT WIDGET  (v3: typing, feedback, search, quick replies)
# ════════════════════════════════════════════════════════════════

class ChatWidget(QWidget):
    message_sent   = Signal(str)
    feedback_given = Signal(str, str, bool)
    quick_reply_clicked = Signal(str)

    def __init__(self, accent="#533483", show_feedback=False, parent=None):
        super().__init__(parent)
        self._accent = accent
        self._show_feedback = show_feedback
        self._typing = False
        self._typing_timer = QTimer(self)
        self._typing_timer.setInterval(350)
        self._typing_dots = 0
        self._typing_timer.timeout.connect(self._tick_typing)
        self._pending_user_msg = ""
        self._last_bot_msg = ""
        self._msg_history = []  # (sender, text, timestamp)
        self._quick_replies = []
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        # Search bar
        search_row = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("🔍 Search in chat…")
        self._search_input.setMaximumHeight(28)
        self._search_input.setStyleSheet(f"""
            QLineEdit{{background:{theme()['bg2']};color:{theme()['text']};
            border:1px solid {theme()['border']};border-radius:4px;padding:2px 8px;font-size:11px;}}
        """)
        self._search_input.textChanged.connect(self._search_chat)
        self._search_btn = QPushButton("✕")
        self._search_btn.setFixedSize(24, 24)
        self._search_btn.setStyleSheet("QPushButton{border:none;color:#888;font-size:12px;}")
        self._search_btn.clicked.connect(lambda: (self._search_input.clear(), self._do_search("")))
        search_row.addWidget(self._search_input, 1)
        search_row.addWidget(self._search_btn)
        lay.addLayout(search_row)

        # Chat display
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setStyleSheet(f"""
            QTextEdit{{background:{theme()['chat_bg']};color:{theme()['text']};
            border:1px solid {theme()['border']};border-radius:8px;padding:10px;font-size:14px;}}
        """)
        lay.addWidget(self.display, 1)

        # Quick replies
        self._quick_row = QHBoxLayout()
        self._quick_row.setSpacing(4)
        self._quick_container = QWidget()
        self._quick_lay = QHBoxLayout(self._quick_container)
        self._quick_lay.setContentsMargins(0, 0, 0, 0)
        self._quick_lay.setSpacing(4)
        self._quick_container.setVisible(False)
        lay.addWidget(self._quick_container)

        # Feedback row
        self._feedback_row = QWidget()
        fl = QHBoxLayout(self._feedback_row)
        fl.setContentsMargins(0, 2, 0, 2)
        fl.addStretch()
        self._lbl_conf = QLabel("")
        self._lbl_conf.setStyleSheet("color:#888;font-size:11px;")
        fl.addWidget(self._lbl_conf)
        for emoji, positive, tip in [("👍", True, "Good response"), ("👎", False, "Poor response")]:
            b = QPushButton(emoji)
            b.setFixedSize(30, 30)
            b.setStyleSheet(
                "QPushButton{border:none;font-size:16px;} QPushButton:hover{background:#2a2a3a;border-radius:4px;}")
            b.setToolTip(tip)
            b.clicked.connect(lambda _, p=positive: self._give_feedback(p))
            fl.addWidget(b)
        self._feedback_row.setVisible(False)
        lay.addWidget(self._feedback_row)

        # Input row
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type your message… (↑↓ for history)")
        self.input.setStyleSheet(f"""
            QLineEdit{{background:{theme()['input']};color:{theme()['text']};
            border:2px solid {theme()['border']};border-radius:20px;padding:10px 20px;font-size:14px;}}
            QLineEdit:focus{{border-color:{self._accent};}}
        """)
        self.input.returnPressed.connect(self._send)
        self._history_idx = -1
        self._input_history = []

        self.btn = QPushButton("➤")
        self.btn.setFixedWidth(44)
        self.btn.setStyleSheet(f"""
            QPushButton{{background:{self._accent};color:white;border:none;
            border-radius:22px;font-size:18px;font-weight:bold;}}
            QPushButton:hover{{background:{self._accent}cc;}}
        """)
        self.btn.clicked.connect(self._send)
        row.addWidget(self.input, 1)
        row.addWidget(self.btn)
        lay.addLayout(row)

    def keyPressEvent(self, ev):
        # Input history navigation
        if ev.key() == Qt.Key_Up and self._input_history:
            self._history_idx = min(self._history_idx + 1, len(self._input_history) - 1)
            self.input.setText(self._input_history[-(self._history_idx + 1)])
        elif ev.key() == Qt.Key_Down:
            self._history_idx = max(self._history_idx - 1, -1)
            self.input.setText(self._input_history[-(self._history_idx + 1)] if self._history_idx >= 0 else "")
        else:
            super().keyPressEvent(ev)

    def _send(self):
        t = self.input.text().strip()
        if t:
            self.add("You", t, is_user=True)
            self._input_history.append(t)
            self._history_idx = -1
            self.input.clear()
            self._pending_user_msg = t
            self._clear_quick_replies()
            self.message_sent.emit(t)

    def _tick_typing(self):
        self._typing_dots = (self._typing_dots % 3) + 1
        dots = "·" * self._typing_dots
        cursor = self.display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.select(cursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertText(f"\n🤖 Bot is typing {dots}")
        self.display.setTextCursor(cursor)
        sb = self.display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def start_typing(self):
        if not self._typing:
            self._typing = True
            self._typing_dots = 0
            self.display.append("")
            self._typing_timer.start()

    def stop_typing_and_respond(self, text, confidence=None):
        self._typing_timer.stop()
        self._typing = False
        cursor = self.display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.select(cursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        self.display.setTextCursor(cursor)
        self.add("Bot", text, is_user=False)
        self._last_bot_msg = text
        if confidence is not None:
            pct = int(confidence * 100)
            color = "#43aa8b" if pct > 60 else "#f0a500" if pct > 30 else "#e94560"
            self._lbl_conf.setStyleSheet(f"color:{color};font-size:11px;font-weight:bold;")
            self._lbl_conf.setText(f"Confidence: {pct}%")
        if self._show_feedback:
            self._feedback_row.setVisible(True)

    def set_quick_replies(self, suggestions):
        self._clear_quick_replies()
        if suggestions:
            self._quick_container.setVisible(True)
            for s in suggestions:
                btn = QPushButton(s[:30])
                btn.setStyleSheet(f"""
                    QPushButton{{background:{theme()['bg2']};color:{theme()['text']};
                    border:1px solid {theme()['border']};border-radius:14px;
                    padding:4px 12px;font-size:11px;}}
                    QPushButton:hover{{border-color:{self._accent};background:{theme()['bg3']};}}
                """)
                btn.clicked.connect(lambda _, txt=s: self._use_quick_reply(txt))
                self._quick_lay.addWidget(btn)
            self._quick_lay.addStretch()

    def _clear_quick_replies(self):
        self._quick_container.setVisible(False)
        while self._quick_lay.count():
            item = self._quick_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _use_quick_reply(self, text):
        self.input.setText(text)
        self._send()

    def _give_feedback(self, positive):
        if self._pending_user_msg and self._last_bot_msg:
            self.feedback_given.emit(self._pending_user_msg, self._last_bot_msg, positive)
            emoji = "👍" if positive else "👎"
            self.add("Feedback", f"{emoji} Marked as {'helpful' if positive else 'unhelpful'}", False)
        self._feedback_row.setVisible(False)

    def add(self, sender, text, is_user=False):
        t = theme()
        c = self._accent if is_user else t["accent2"]
        a = "right" if is_user else "left"
        ts = datetime.now().strftime("%H:%M:%S")
        self._msg_history.append((sender, text, ts))
        self.display.append(
            f'<div style="margin:4px 0;text-align:{a};">'
            f'<span style="color:#555;font-size:9px;">[{ts}]</span> '
            f'<span style="color:{c};font-weight:bold;">{sender}:</span> '
            f'<span style="color:{t["text"]};">{text}</span></div>')
        sb = self.display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _search_chat(self, text):
        self._do_search(text)

    def _do_search(self, text):
        # Highlight search results by re-rendering
        if not text.strip():
            # Rebuild without highlights
            saved = self._msg_history[:]
            self.display.clear()
            for sender, msg, ts in saved:
                is_user = sender == "You"
                t = theme()
                c = self._accent if is_user else t["accent2"]
                a = "right" if is_user else "left"
                self.display.append(
                    f'<div style="margin:4px 0;text-align:{a};">'
                    f'<span style="color:#555;font-size:9px;">[{ts}]</span> '
                    f'<span style="color:{c};font-weight:bold;">{sender}:</span> '
                    f'<span style="color:{t["text"]};">{msg}</span></div>')
            return
        # Simple: just find and scroll
        cursor = self.display.document().find(text)
        if not cursor.isNull():
            self.display.setTextCursor(cursor)
            self.display.ensureCursorVisible()

    def clear_chat(self):
        self.display.clear()
        self._msg_history.clear()
        self._feedback_row.setVisible(False)
        self._clear_quick_replies()

    def export_html(self, path):
        t = theme()
        html = f"""<html><head><meta charset="utf-8"><style>
            body{{background:{t['bg']};color:{t['text']};font-family:sans-serif;padding:20px;max-width:800px;margin:0 auto;}}
            .msg{{margin:8px 0;padding:8px 12px;border-radius:8px;}}
            .user{{background:{t['bg2']};text-align:right;}}
            .bot{{background:{t['bg3']};text-align:left;}}
            .ts{{color:#555;font-size:10px;}}
            h2{{color:{t['accent']};}}
            </style></head><body>
            <h2>Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}</h2>
            {self.display.toHtml()}</body></html>"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)

    def export_json(self, path):
        data = [{"sender": s, "text": t, "time": ts} for s, t, ts in self._msg_history]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════
#  HELPER WIDGETS
# ════════════════════════════════════════════════════════════════

def styled_group(title, accent):
    g = QGroupBox(title)
    g.setStyleSheet(f"""
        QGroupBox{{color:{accent};font-weight:bold;border:2px solid {theme()['border']};
                   border-radius:8px;margin-top:12px;padding-top:18px;}}
        QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 6px;}}""")
    return g

def styled_btn(text, bg, slot=None, tooltip=""):
    b = QPushButton(text)
    b.setStyleSheet(f"""
        QPushButton{{background:{bg};color:white;border:none;border-radius:5px;
                     padding:8px 14px;font-weight:bold;}}
        QPushButton:hover{{background:{bg}cc;}}
        QPushButton:disabled{{background:#333;color:#666;}}""")
    if slot:
        b.clicked.connect(slot)
    if tooltip:
        b.setToolTip(tooltip)
    return b

class DropArea(QTextEdit):
    """Text edit that accepts drag-and-drop text files."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._drop_callback = None

    def set_drop_callback(self, cb):
        self._drop_callback = cb

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls and urls[0].toLocalFile().endswith('.txt'):
            path = urls[0].toLocalFile()
            with open(path, encoding='utf-8', errors='ignore') as f:
                self.setPlainText(f.read())
            if self._drop_callback:
                self._drop_callback(path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


# ════════════════════════════════════════════════════════════════
#  MDP INTERACTIVE EXPLORER DIALOG
# ════════════════════════════════════════════════════════════════

class MDPExplorerDialog(QDialog):
    """Step through MDP states interactively."""

    def __init__(self, engine: MDPChatbot, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.current_sid = 0
        self.setWindowTitle("MDP State Explorer")
        self.setMinimumSize(600, 500)
        self.setStyleSheet(f"""
            QDialog{{background:{theme()['bg']};color:{theme()['text']};}}
            QLabel{{color:{theme()['text']};}}
            QPushButton{{background:{theme()['bg2']};color:{theme()['text']};
                         border:1px solid {theme()['border']};border-radius:4px;padding:6px 12px;}}
            QPushButton:hover{{border-color:{ACCENT_MDP};}}
            QTextEdit{{background:{theme()['bg2']};color:{theme()['text']};
                       border:1px solid {theme()['border']};border-radius:4px;padding:5px;}}
        """)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        # Current state
        self.lbl_state = QLabel("State: S0")
        self.lbl_state.setStyleSheet(f"color:{ACCENT_MDP};font-size:16px;font-weight:bold;")
        lay.addWidget(self.lbl_state)

        self.lbl_text = QLabel("")
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setStyleSheet("font-size:14px;padding:8px;background:#16213e;border-radius:6px;")
        lay.addWidget(self.lbl_text)

        # Value info
        info_row = QHBoxLayout()
        self.lbl_value = QLabel("V(s) = 0.000")
        self.lbl_value.setStyleSheet("font-size:13px;font-weight:bold;color:#43aa8b;")
        info_row.addWidget(self.lbl_value)
        self.lbl_policy = QLabel("Policy → S?")
        self.lbl_policy.setStyleSheet("font-size:13px;font-weight:bold;color:#f0a500;")
        info_row.addWidget(self.lbl_policy)
        lay.addLayout(info_row)

        # Available actions
        lay.addWidget(QLabel("Available actions (transitions):"))
        self.actions_list = QTextEdit()
        self.actions_list.setReadOnly(True)
        self.actions_list.setMaximumHeight(150)
        lay.addWidget(self.actions_list)

        # Navigation
        nav = QHBoxLayout()
        nav.addWidget(styled_btn("← Previous State", ACCENT_MDP, self._prev))
        nav.addWidget(styled_btn("Random State", ACCENT_MDP, self._random))
        nav.addWidget(styled_btn("Follow Policy →", ACCENT_MDP, self._follow_policy))
        nav.addWidget(styled_btn("Next State →", ACCENT_MDP, self._next))
        lay.addLayout(nav)

        # Jump to state
        jump_row = QHBoxLayout()
        self.sp_jump = QSpinBox()
        self.sp_jump.setRange(0, max(0, len(self.engine.states) - 1))
        jump_row.addWidget(QLabel("Jump to state:"))
        jump_row.addWidget(self.sp_jump)
        jump_row.addWidget(styled_btn("Go", ACCENT_MDP, self._jump))
        lay.addLayout(jump_row)

        self._update_display()

    def _update_display(self):
        sid = self.current_sid
        self.lbl_state.setText(f"State: S{sid}")
        if sid in self.engine.states:
            info = self.engine.states[sid]
            self.lbl_text.setText(f'"{info["text"]}"')
        else:
            self.lbl_text.setText("(unknown state)")

        v = self.engine.values.get(sid, 0)
        self.lbl_value.setText(f"V(s) = {v:.4f}")

        pol = self.engine.policy.get(sid)
        self.lbl_policy.setText(f"Policy → S{pol}" if pol is not None else "Policy → (none)")

        self.actions_list.clear()
        if sid in self.engine.transitions:
            for aid, trans in self.engine.transitions[sid].items():
                reward = self.engine.rewards[sid].get(aid, 0)
                for nsid, prob in trans:
                    txt = self.engine.states[nsid]["text"][:40] if nsid in self.engine.states else "?"
                    marker = " ◄ POLICY" if pol == nsid else ""
                    self.actions_list.append(
                        f"  Action {aid}: → S{nsid}  P={prob:.3f}  R={reward:.3f}  \"{txt}\"{marker}")

    def _prev(self):
        self.current_sid = max(0, self.current_sid - 1)
        self._update_display()

    def _next(self):
        self.current_sid = min(len(self.engine.states) - 1, self.current_sid + 1)
        self._update_display()

    def _random(self):
        self.current_sid = random.choice(list(self.engine.states.keys()))
        self._update_display()

    def _follow_policy(self):
        if self.current_sid in self.engine.policy:
            self.current_sid = self.engine.policy[self.current_sid]
            self._update_display()

    def _jump(self):
        self.current_sid = self.sp_jump.value()
        self._update_display()


# ════════════════════════════════════════════════════════════════
#  MODEL MANAGER
# ════════════════════════════════════════════════════════════════

class ModelRecord:
    def __init__(self, name, engine_type, engine, created=None):
        self.name = name
        self.engine_type = engine_type  # "mc" or "mdp"
        self.engine = engine
        self.created = created or datetime.now().strftime("%Y-%m-%d %H:%M")
        self.id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]


class ModelManagerDialog(QDialog):
    """Manage multiple trained models."""

    def __init__(self, models: dict, parent=None):
        super().__init__(parent)
        self.models = models  # id → ModelRecord
        self.selected_id = None
        self.setWindowTitle("Model Manager")
        self.setMinimumSize(500, 400)
        self.setStyleSheet(f"""
            QDialog{{background:{theme()['bg']};color:{theme()['text']};}}
            QListWidget{{background:{theme()['bg2']};color:{theme()['text']};
                         border:1px solid {theme()['border']};border-radius:4px;}}
            QListWidget::item:selected{{background:{ACCENT_MGR};}}
            QLabel{{color:{theme()['text']};}}
            QPushButton{{background:{theme()['bg2']};color:{theme()['text']};
                         border:1px solid {theme()['border']};border-radius:4px;padding:6px 14px;}}
            QPushButton:hover{{border-color:{ACCENT_MGR};}}
        """)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel("📦 Saved Models"))
        self.list_w = QListWidget()
        for mid, rec in self.models.items():
            icon = "🔗" if rec.engine_type == "mc" else "🎯"
            item = QListWidgetItem(f"{icon} {rec.name}  ({rec.engine_type.upper()})  — {rec.created}")
            item.setData(Qt.UserRole, mid)
            self.list_w.addItem(item)
        self.list_w.itemDoubleClicked.connect(self._select)
        lay.addWidget(self.list_w, 1)

        # Info
        self.lbl_info = QLabel("Select a model to see details")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("padding:8px;background:#16213e;border-radius:4px;")
        lay.addWidget(self.lbl_info)
        self.list_w.currentItemChanged.connect(self._info_changed)

        # Buttons
        row = QHBoxLayout()
        row.addWidget(styled_btn("Load", ACCENT_MGR, self._select))
        row.addWidget(styled_btn("Rename", "#333", self._rename))
        row.addWidget(styled_btn("Delete", "#e94560", self._delete))
        row.addWidget(styled_btn("Export", "#333", self._export))
        row.addStretch()
        row.addWidget(styled_btn("Cancel", "#333", self.reject))
        lay.addLayout(row)

    def _info_changed(self, current, _):
        if current:
            mid = current.data(Qt.UserRole)
            rec = self.models.get(mid)
            if rec:
                stats = rec.engine.get_stats()
                info_lines = [f"Name: {rec.name}", f"Type: {rec.engine_type.upper()}",
                              f"Created: {rec.created}"]
                for k, v in stats.items():
                    info_lines.append(f"{k}: {v}")
                self.lbl_info.setText("\n".join(info_lines))

    def _select(self):
        item = self.list_w.currentItem()
        if item:
            self.selected_id = item.data(Qt.UserRole)
            self.accept()

    def _rename(self):
        item = self.list_w.currentItem()
        if item:
            mid = item.data(Qt.UserRole)
            rec = self.models[mid]
            name, ok = QInputDialog.getText(self, "Rename", "New name:", text=rec.name)
            if ok and name:
                rec.name = name
                self.list_w.clear()
                for m_id, r in self.models.items():
                    icon = "🔗" if r.engine_type == "mc" else "🎯"
                    i = QListWidgetItem(f"{icon} {r.name}  ({r.engine_type.upper()})  — {r.created}")
                    i.setData(Qt.UserRole, m_id)
                    self.list_w.addItem(i)

    def _delete(self):
        item = self.list_w.currentItem()
        if item:
            mid = item.data(Qt.UserRole)
            if QMessageBox.question(self, "Delete", f"Delete '{self.models[mid].name}'?") == QMessageBox.Yes:
                del self.models[mid]
                self.list_w.takeItem(self.list_w.row(item))

    def _export(self):
        item = self.list_w.currentItem()
        if item:
            mid = item.data(Qt.UserRole)
            rec = self.models[mid]
            path, _ = QFileDialog.getSaveFileName(self, "Export Model",
                                                   f"{rec.name}.pkl", "Pickle (*.pkl)")
            if path:
                with open(path, 'wb') as f:
                    pickle.dump(rec.engine, f)


# ════════════════════════════════════════════════════════════════
#  MARKOV CHAIN TAB
# ════════════════════════════════════════════════════════════════

class MarkovChainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = MarkovChain(order=2)
        self._context = []  # multi-turn context
        self._saved_models = {}  # id → ModelRecord
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.setInterval(60000)  # 1 min
        self._auto_save_timer.timeout.connect(self._auto_save)
        self._build_ui()
        self._auto_save_timer.start()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # LEFT
        left = QWidget(); left.setFixedWidth(340)
        ll = QVBoxLayout(left); ll.setContentsMargins(4, 4, 4, 4)

        cg = styled_group("⚙  Configuration", ACCENT_MC)
        cl = QFormLayout()

        self.sp_order = QSpinBox(); self.sp_order.setRange(1, 5); self.sp_order.setValue(2)
        self.sp_order.setToolTip("N-gram order: how many previous words the chain considers")
        cl.addRow("Chain order:", self.sp_order)

        self.sp_max = QSpinBox(); self.sp_max.setRange(5, 500); self.sp_max.setValue(60)
        cl.addRow("Max words:", self.sp_max)

        self.sp_temp = QDoubleSpinBox(); self.sp_temp.setRange(0.1, 3.0)
        self.sp_temp.setValue(1.0); self.sp_temp.setSingleStep(0.1)
        self.sp_temp.setToolTip("<0.5=focused, 1.0=balanced, >1.5=creative")
        cl.addRow("Temperature:", self.sp_temp)

        # Response mode
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Balanced", "Creative", "Precise", "Completion"])
        self.cmb_mode.setToolTip("Balanced=default, Creative=high temp, Precise=low temp, Completion=continue seed")
        self.cmb_mode.setStyleSheet(
            f"background:{theme()['bg2']};color:{theme()['text']};border:1px solid {theme()['border']};border-radius:4px;padding:4px;")
        cl.addRow("Mode:", self.cmb_mode)

        # Batch count
        self.sp_batch = QSpinBox(); self.sp_batch.setRange(1, 5); self.sp_batch.setValue(1)
        self.sp_batch.setToolTip("Generate multiple responses and pick the best")
        cl.addRow("Batch size:", self.sp_batch)

        # Preprocessing
        self.chk_lower = QCheckBox("Lowercase"); self.chk_lower.setChecked(True)
        self.chk_stops = QCheckBox("Remove stopwords")
        pp_row = QHBoxLayout(); pp_row.addWidget(self.chk_lower); pp_row.addWidget(self.chk_stops)
        cl.addRow("Preprocess:", pp_row)

        # Real-time learning
        self.chk_learn = QCheckBox("Learn from chat")
        self.chk_learn.setChecked(True)
        self.chk_learn.setToolTip("Incrementally learn from conversation messages")
        self.chk_learn.setStyleSheet("color:#ccc;")
        cl.addRow(self.chk_learn)

        cg.setLayout(cl); ll.addWidget(cg)

        # Training
        tg = styled_group("📚  Training Data", ACCENT_MC)
        tl = QVBoxLayout()
        self.train_text = DropArea()
        self.train_text.setPlaceholderText("Paste text here… or drag & drop a .txt file!")
        self.train_text.setMaximumHeight(130)
        self.train_text.setStyleSheet(
            f"QTextEdit{{background:{theme()['bg2']};color:{theme()['text']};border:2px dashed {theme()['border']};border-radius:4px;padding:5px;}}")
        self.train_text.set_drop_callback(lambda p: ToastManager.show(f"Loaded: {os.path.basename(p)}", "success"))
        tl.addWidget(self.train_text)
        brow = QHBoxLayout()
        brow.addWidget(styled_btn("Train", ACCENT_MC, self._train, "Train Markov Chain (Ctrl+T)"))
        brow.addWidget(styled_btn("Sample", theme()['border'], self._sample, "Load sample text"))
        brow.addWidget(styled_btn("File…", theme()['border'], self._load_file, "Load from .txt file"))
        tl.addLayout(brow)
        srow = QHBoxLayout()
        srow.addWidget(styled_btn("💾 Save", "#1a6b3c", self._save_model, "Save model"))
        srow.addWidget(styled_btn("📂 Load", "#1a6b3c", self._load_model, "Load model"))
        srow.addWidget(styled_btn("📦 Manager", ACCENT_MGR, self._open_manager, "Manage saved models"))
        tl.addLayout(srow)
        tg.setLayout(tl); ll.addWidget(tg)

        # Stats + Perplexity
        sg = styled_group("📊  Statistics", ACCENT_MC)
        sl = QVBoxLayout()
        self.lbl_stats = QLabel("No model trained yet.")
        self.lbl_stats.setStyleSheet("color:#ccc;"); self.lbl_stats.setWordWrap(True)
        sl.addWidget(self.lbl_stats)

        # Perplexity test
        perp_row = QHBoxLayout()
        self.perp_input = QLineEdit(); self.perp_input.setPlaceholderText("Test perplexity…")
        self.perp_input.setStyleSheet(
            f"background:{theme()['bg2']};color:{theme()['text']};border:1px solid {theme()['border']};border-radius:4px;padding:4px;")
        perp_row.addWidget(self.perp_input, 1)
        perp_row.addWidget(styled_btn("Calc", ACCENT_MC, self._calc_perplexity, "Compute perplexity"))
        sl.addLayout(perp_row)
        self.lbl_perplexity = QLabel("Perplexity: —")
        self.lbl_perplexity.setStyleSheet("color:#f0a500;font-size:11px;")
        sl.addWidget(self.lbl_perplexity)
        sg.setLayout(sl); ll.addWidget(sg)

        ll.addStretch(); root.addWidget(left)

        # CENTRE — chat
        centre = QWidget(); cl2 = QVBoxLayout(centre); cl2.setContentsMargins(0, 0, 0, 0)
        title_row = QHBoxLayout()
        title = QLabel("🔗  Markov Chain Chatbot")
        title.setStyleSheet(f"color:{ACCENT_MC};font-size:18px;font-weight:bold;padding:4px;")
        title.setAlignment(Qt.AlignCenter)
        title_row.addWidget(title, 1)
        # Context indicator
        self.lbl_context = QLabel("Context: 0 msgs")
        self.lbl_context.setStyleSheet("color:#666;font-size:10px;")
        title_row.addWidget(self.lbl_context)
        cl2.addLayout(title_row)
        self.chat = ChatWidget(accent=ACCENT_MC)
        self.chat.message_sent.connect(self._on_msg)
        cl2.addWidget(self.chat, 1)

        # Export row
        erow = QHBoxLayout()
        erow.addWidget(styled_btn("📋 Export HTML", theme()['bg2'], self._export_html, "Export chat as HTML"))
        erow.addWidget(styled_btn("📄 Export JSON", theme()['bg2'], self._export_json, "Export chat as JSON"))
        erow.addWidget(styled_btn("🗑 Clear Chat", "#4a2020", self.chat.clear_chat, "Clear chat history"))
        erow.addStretch()
        cl2.addLayout(erow)
        root.addWidget(centre, 1)

        # RIGHT — graph + heatmap
        right = QWidget(); right.setFixedWidth(400)
        rl = QVBoxLayout(right); rl.setContentsMargins(4, 4, 4, 4)
        gtitle = QLabel("🕸  Transition Graph")
        gtitle.setStyleSheet(f"color:{ACCENT_MC};font-size:13px;font-weight:bold;")
        gtitle.setAlignment(Qt.AlignCenter); rl.addWidget(gtitle)
        hint = QLabel("Scroll=zoom · Right-drag=pan · Double-click=force layout")
        hint.setStyleSheet("color:#555;font-size:8px;"); hint.setAlignment(Qt.AlignCenter)
        rl.addWidget(hint)
        self.graph = GraphWidget(); rl.addWidget(self.graph, 1)
        rrow = QHBoxLayout()
        rrow.addWidget(styled_btn("🔄 Refresh", ACCENT_MC, self._refresh_graph))
        rrow.addWidget(styled_btn("⚡ Force Layout", ACCENT_MC, self.graph.start_force_layout))
        rl.addLayout(rrow)

        # Heatmap
        rl.addWidget(QLabel("🌡  Transition Heatmap:"))
        self.heatmap = HeatmapWidget(); self.heatmap.setMinimumHeight(200)
        rl.addWidget(self.heatmap)
        root.addWidget(right)

    # ── slots ───────────────────────────────────────────────────
    def _train(self):
        txt = self.train_text.toPlainText().strip()
        if not txt:
            ToastManager.show("Enter training text first!", "warning"); return
        pp = PreprocessConfig(lowercase=self.chk_lower.isChecked(),
                              remove_stops=self.chk_stops.isChecked())
        self.engine = MarkovChain(order=self.sp_order.value(), preprocess=pp)
        self.engine.train(txt)
        self._context.clear()
        s = self.engine.get_stats()
        self.lbl_stats.setText(
            f"✅ Trained!\nOrder: {s['order']}\nStates: {s['states']}\n"
            f"Transitions: {s['transitions']}\nVocabulary: {s['vocabulary']}\n"
            f"Raw chars: {s['raw_chars']}")
        self.chat.add("System", "Model trained — start chatting!", False)
        ToastManager.show("Markov Chain model trained!", "success")
        self._refresh_graph()
        self._refresh_heatmap()

    def _sample(self):
        self.train_text.setPlainText(SAMPLE_TEXT)
        ToastManager.show("Sample text loaded", "info")

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text (*.txt);;All (*)")
        if path:
            with open(path, encoding='utf-8', errors='ignore') as f:
                self.train_text.setPlainText(f.read())
            ToastManager.show(f"Loaded: {os.path.basename(path)}", "success")

    def _save_model(self):
        if not self.engine.trained:
            ToastManager.show("Train a model first!", "warning"); return
        name, ok = QInputDialog.getText(self, "Save Model", "Model name:", text=f"MC_order{self.engine.order}")
        if ok and name:
            rec = ModelRecord(name, "mc", self.engine)
            self._saved_models[rec.id] = rec
            path, _ = QFileDialog.getSaveFileName(self, "Save", f"{name}.pkl", "Pickle (*.pkl)")
            if path:
                with open(path, 'wb') as f:
                    pickle.dump(self.engine, f)
            ToastManager.show(f"Model '{name}' saved!", "success")

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Pickle (*.pkl)")
        if path:
            try:
                with open(path, 'rb') as f:
                    self.engine = pickle.load(f)
                s = self.engine.get_stats()
                self.lbl_stats.setText(
                    f"✅ Loaded!\nOrder: {s['order']}\nStates: {s['states']}\n"
                    f"Transitions: {s['transitions']}\nVocabulary: {s['vocabulary']}")
                self.sp_order.setValue(self.engine.order)
                self.chat.add("System", f"Model loaded from {os.path.basename(path)}", False)
                self._refresh_graph()
                self._refresh_heatmap()
                ToastManager.show("Model loaded!", "success")
            except Exception as e:
                ToastManager.show(f"Load failed: {e}", "error")

    def _open_manager(self):
        if not self._saved_models:
            ToastManager.show("No saved models yet. Save a model first.", "info"); return
        dlg = ModelManagerDialog(self._saved_models, self)
        if dlg.exec() == QDialog.Accepted and dlg.selected_id:
            rec = self._saved_models[dlg.selected_id]
            self.engine = rec.engine
            s = self.engine.get_stats()
            self.lbl_stats.setText(f"✅ Loaded '{rec.name}'!\n" + "\n".join(
                f"{k}: {v}" for k, v in s.items()))
            self.chat.add("System", f"Switched to model '{rec.name}'", False)
            self._refresh_graph()
            self._refresh_heatmap()

    def _auto_save(self):
        """Auto-save to temp file if model is trained."""
        if self.engine.trained:
            tmp = os.path.join(os.path.dirname(__file__) or ".", ".mc_autosave.pkl")
            try:
                with open(tmp, 'wb') as f:
                    pickle.dump(self.engine, f)
            except Exception:
                pass

    def _on_msg(self, text):
        if not self.engine.trained:
            self.chat.add("Bot", "Train the model first!", False); return
        self._context.append(text)
        self.lbl_context.setText(f"Context: {len(self._context)} msgs")

        # Real-time learning
        if self.chk_learn.isChecked():
            self.engine.learn_from_conversation(text)

        self.chat.start_typing()
        delay = random.randint(200, 600)
        QTimer.singleShot(delay, lambda: self._respond(text))

    def _respond(self, text):
        mode = self.cmb_mode.currentText()
        if mode == "Creative":
            temp = 1.6
        elif mode == "Precise":
            temp = 0.3
        else:
            temp = self.sp_temp.value()

        completion = (mode == "Completion")
        batch = self.sp_batch.value()

        if batch > 1:
            results = self.engine.generate_batch(seed=text, max_words=self.sp_max.value(),
                                                  temperature=temp, context=self._context,
                                                  count=batch)
            resp, conf = results[0]
        else:
            resp, conf = self.engine.generate(seed=text, max_words=self.sp_max.value(),
                                               temperature=temp, context=self._context,
                                               completion_mode=completion)

        if not resp.strip() or resp.startswith("⚠") or resp.startswith("🤔"):
            resp, conf = self.engine.generate(max_words=self.sp_max.value(),
                                               temperature=temp, context=self._context)
        self.chat.stop_typing_and_respond(resp, conf)

        # Quick replies
        if self.engine.trained:
            suggestions = self.engine.suggest_replies(context=self._context, count=3)
            self.chat.set_quick_replies(suggestions)

        self._context.append(resp)
        self.lbl_context.setText(f"Context: {len(self._context)} msgs")

    def _calc_perplexity(self):
        text = self.perp_input.text().strip()
        if not text:
            ToastManager.show("Enter text to test perplexity", "warning"); return
        if not self.engine.trained:
            ToastManager.show("Train a model first!", "warning"); return
        ppl = self.engine.compute_perplexity(text)
        if ppl == float('inf'):
            self.lbl_perplexity.setText("Perplexity: ∞ (unseen n-grams)")
        else:
            color = "#43aa8b" if ppl < 50 else "#f0a500" if ppl < 200 else "#e94560"
            self.lbl_perplexity.setStyleSheet(f"color:{color};font-size:11px;font-weight:bold;")
            self.lbl_perplexity.setText(f"Perplexity: {ppl:.2f}")

    def _export_html(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export HTML", "chat_mc.html", "HTML (*.html)")
        if path:
            self.chat.export_html(path)
            ToastManager.show(f"Exported to {os.path.basename(path)}", "success")

    def _export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", "chat_mc.json", "JSON (*.json)")
        if path:
            self.chat.export_json(path)
            ToastManager.show(f"Exported to {os.path.basename(path)}", "success")

    def _refresh_graph(self):
        if self.engine.trained:
            self.graph.set_data(*self.engine.get_graph_data(35))

    def _refresh_heatmap(self):
        if self.engine.trained:
            labels, matrix = self.engine.get_transition_matrix(12)
            self.heatmap.set_data(labels, matrix)


# ════════════════════════════════════════════════════════════════
#  MDP TAB
# ════════════════════════════════════════════════════════════════

class MDPTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = MDPChatbot()
        self._context = []
        self._saved_models = {}
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # LEFT
        left = QWidget(); left.setFixedWidth(340)
        ll = QVBoxLayout(left); ll.setContentsMargins(4, 4, 4, 4)

        cg = styled_group("⚙  MDP Configuration", ACCENT_MDP)
        cl = QFormLayout()
        self.sp_gamma = QDoubleSpinBox(); self.sp_gamma.setRange(0.0, 1.0)
        self.sp_gamma.setValue(0.9); self.sp_gamma.setSingleStep(0.05)
        self.sp_gamma.setToolTip("Discount factor: how much future rewards matter")
        cl.addRow("Discount γ:", self.sp_gamma)

        self.sp_iter = QSpinBox(); self.sp_iter.setRange(10, 5000); self.sp_iter.setValue(200)
        cl.addRow("Max iterations:", self.sp_iter)

        self.sp_theta = QDoubleSpinBox(); self.sp_theta.setRange(0.00001, 1.0)
        self.sp_theta.setValue(0.001); self.sp_theta.setDecimals(5); self.sp_theta.setSingleStep(0.001)
        self.sp_theta.setToolTip("Convergence threshold δ")
        cl.addRow("Threshold θ:", self.sp_theta)

        self.cmb_solver = QComboBox()
        self.cmb_solver.addItems(["Value Iteration", "Q-Learning"])
        self.cmb_solver.setStyleSheet(
            f"background:{theme()['bg2']};color:{theme()['text']};border:1px solid {theme()['border']};border-radius:4px;padding:4px;")
        self.cmb_solver.currentIndexChanged.connect(self._solver_changed)
        cl.addRow("Solver:", self.cmb_solver)

        self._ql_widget = QWidget()
        ql = QFormLayout(self._ql_widget); ql.setContentsMargins(0, 0, 0, 0)
        self.sp_alpha = QDoubleSpinBox(); self.sp_alpha.setRange(0.01, 1.0)
        self.sp_alpha.setValue(0.1); self.sp_alpha.setSingleStep(0.05)
        ql.addRow("Learning rate α:", self.sp_alpha)
        self.sp_epsilon = QDoubleSpinBox(); self.sp_epsilon.setRange(0.0, 1.0)
        self.sp_epsilon.setValue(0.2); self.sp_epsilon.setSingleStep(0.05)
        ql.addRow("Exploration ε:", self.sp_epsilon)
        self.sp_episodes = QSpinBox(); self.sp_episodes.setRange(100, 10000)
        self.sp_episodes.setValue(500)
        ql.addRow("Episodes:", self.sp_episodes)
        self._ql_widget.setVisible(False)
        cl.addRow(self._ql_widget)

        self.cb_live = QCheckBox("Live log"); self.cb_live.setChecked(True)
        self.cb_live.setStyleSheet("color:#ccc;")
        cl.addRow(self.cb_live)

        self.chk_lower_mdp = QCheckBox("Lowercase"); self.chk_lower_mdp.setChecked(True)
        self.chk_learn_mdp = QCheckBox("Learn from chat"); self.chk_learn_mdp.setChecked(True)
        self.chk_learn_mdp.setStyleSheet("color:#ccc;")
        pp_row = QHBoxLayout()
        pp_row.addWidget(self.chk_lower_mdp); pp_row.addWidget(self.chk_learn_mdp)
        cl.addRow("Preprocess:", pp_row)
        cg.setLayout(cl); ll.addWidget(cg)

        tg = styled_group("📚  Training Data", ACCENT_MDP)
        tl = QVBoxLayout()
        self.train_text = DropArea()
        self.train_text.setPlaceholderText("Paste text here… or drag & drop a .txt file!")
        self.train_text.setMaximumHeight(100)
        self.train_text.setStyleSheet(
            f"QTextEdit{{background:{theme()['bg2']};color:{theme()['text']};border:2px dashed {theme()['border']};border-radius:4px;padding:5px;}}")
        tl.addWidget(self.train_text)
        brow = QHBoxLayout()
        brow.addWidget(styled_btn("Train & Solve", ACCENT_MDP, self._train))
        brow.addWidget(styled_btn("Sample", theme()['border'], self._sample))
        brow.addWidget(styled_btn("File…", theme()['border'], self._load_file))
        tl.addLayout(brow)
        srow = QHBoxLayout()
        srow.addWidget(styled_btn("💾 Save", "#1a6b3c", self._save_model))
        srow.addWidget(styled_btn("📂 Load", "#1a6b3c", self._load_model))
        srow.addWidget(styled_btn("📦 Manager", ACCENT_MGR, self._open_manager))
        tl.addLayout(srow)
        tg.setLayout(tl); ll.addWidget(tg)

        vg = styled_group("🔁  Solver Progress", ACCENT_MDP)
        vl = QVBoxLayout()
        self.progress = QProgressBar()
        self.progress.setStyleSheet(f"""
            QProgressBar{{background:{theme()['bg2']};border:1px solid {theme()['border']};
            border-radius:4px;text-align:center;color:white;}}
            QProgressBar::chunk{{background:{ACCENT_MDP};border-radius:3px;}}""")
        vl.addWidget(self.progress)
        self.vi_log = QTextEdit(); self.vi_log.setReadOnly(True); self.vi_log.setMaximumHeight(70)
        self.vi_log.setStyleSheet(
            f"QTextEdit{{background:{theme()['bg2']};color:#90e0ef;border:1px solid {theme()['border']};"
            "border-radius:4px;padding:4px;font-family:Consolas,monospace;font-size:10px;}")
        vl.addWidget(self.vi_log)
        vg.setLayout(vl); ll.addWidget(vg)

        sg = styled_group("📊  Statistics", ACCENT_MDP)
        sl2 = QVBoxLayout()
        self.lbl_stats = QLabel("No model trained yet.")
        self.lbl_stats.setStyleSheet("color:#ccc;"); self.lbl_stats.setWordWrap(True)
        sl2.addWidget(self.lbl_stats)
        sg.setLayout(sl2); ll.addWidget(sg)

        ll.addStretch(); root.addWidget(left)

        # CENTRE
        centre = QWidget(); cl2 = QVBoxLayout(centre); cl2.setContentsMargins(0, 0, 0, 0)
        title_row = QHBoxLayout()
        title = QLabel("🎯  MDP Chatbot")
        title.setStyleSheet(f"color:{ACCENT_MDP};font-size:18px;font-weight:bold;padding:4px;")
        title.setAlignment(Qt.AlignCenter); title_row.addWidget(title, 1)
        self.lbl_context_mdp = QLabel("Context: 0 msgs")
        self.lbl_context_mdp.setStyleSheet("color:#666;font-size:10px;")
        title_row.addWidget(self.lbl_context_mdp)
        cl2.addLayout(title_row)

        self.chat = ChatWidget(accent=ACCENT_MDP, show_feedback=True)
        self.chat.message_sent.connect(self._on_msg)
        self.chat.feedback_given.connect(self._on_feedback)
        cl2.addWidget(self.chat, 1)

        self.conv_chart = ConvergenceChart(); self.conv_chart.setFixedHeight(160)
        cl2.addWidget(self.conv_chart)

        erow = QHBoxLayout()
        erow.addWidget(styled_btn("📋 Export HTML", theme()['bg2'], self._export_html))
        erow.addWidget(styled_btn("📄 Export JSON", theme()['bg2'], self._export_json))
        erow.addWidget(styled_btn("🔍 Explore States", ACCENT_MDP, self._open_explorer,
                                   "Interactive MDP state explorer"))
        erow.addWidget(styled_btn("🗑 Clear", "#4a2020", self.chat.clear_chat))
        erow.addStretch()
        cl2.addLayout(erow)
        root.addWidget(centre, 1)

        # RIGHT
        right = QWidget(); right.setFixedWidth(400)
        rl = QVBoxLayout(right); rl.setContentsMargins(4, 4, 4, 4)
        gtitle = QLabel("🕸  Policy Graph")
        gtitle.setStyleSheet(f"color:{ACCENT_MDP};font-size:13px;font-weight:bold;")
        gtitle.setAlignment(Qt.AlignCenter); rl.addWidget(gtitle)
        hint = QLabel("Scroll=zoom · Right-drag=pan · Double-click=force layout")
        hint.setStyleSheet("color:#555;font-size:8px;"); hint.setAlignment(Qt.AlignCenter)
        rl.addWidget(hint)
        self.graph = GraphWidget(); rl.addWidget(self.graph, 1)
        self.val_table = QTextEdit(); self.val_table.setReadOnly(True)
        self.val_table.setMaximumHeight(140)
        self.val_table.setStyleSheet(
            f"QTextEdit{{background:{theme()['bg2']};color:#90e0ef;border:1px solid {theme()['border']};"
            "border-radius:4px;padding:4px;font-family:Consolas,monospace;font-size:10px;}")
        rl.addWidget(self.val_table)
        rrow = QHBoxLayout()
        rrow.addWidget(styled_btn("🔄 Refresh", ACCENT_MDP, self._refresh_graph))
        rrow.addWidget(styled_btn("⚡ Force Layout", ACCENT_MDP, self.graph.start_force_layout))
        rl.addLayout(rrow)
        root.addWidget(right)

    def _solver_changed(self, idx):
        self._ql_widget.setVisible(idx == 1)

    def _train(self):
        txt = self.train_text.toPlainText().strip()
        if not txt:
            ToastManager.show("Enter training text first!", "warning"); return
        pp = PreprocessConfig(lowercase=self.chk_lower_mdp.isChecked())
        self.engine = MDPChatbot(gamma=self.sp_gamma.value(),
                                  theta=self.sp_theta.value(), preprocess=pp)
        self.engine.train(txt)
        self._context.clear()

        self.vi_log.clear(); self.conv_chart.clear()
        timer = QElapsedTimer(); timer.start()
        use_vi = self.cmb_solver.currentIndex() == 0

        if use_vi:
            iters = self.sp_iter.value()
            self.progress.setRange(0, iters); self.progress.setValue(0)
            for i in range(iters):
                delta = self.engine.value_iteration_step()
                self.engine.convergence_history.append((i + 1, delta))
                self.progress.setValue(i + 1)
                if self.cb_live.isChecked() and (i < 5 or (i + 1) % 20 == 0):
                    self.vi_log.append(f"iter {i + 1:>4d}  δ={delta:.8f}")
                if delta < self.sp_theta.value():
                    self.vi_log.append(f"✅ Converged at iter {i + 1}  (δ={delta:.8f})")
                    break
                if i % 5 == 0:
                    QApplication.processEvents()
            self.engine.extract_policy()
        else:
            episodes = self.sp_episodes.value()
            self.progress.setRange(0, episodes); self.progress.setValue(0)
            self.engine.run_q_learning(
                episodes=episodes, alpha=self.sp_alpha.value(),
                epsilon=self.sp_epsilon.value(),
                callback=lambda ep, delta: self._ql_callback(ep, delta, episodes))

        elapsed = timer.elapsed()
        self.conv_chart.set_data(self.engine.convergence_history)
        s = self.engine.get_stats()
        solver = "Value Iteration" if use_vi else "Q-Learning"
        self.lbl_stats.setText(
            f"✅ {solver} solved in {elapsed}ms\n"
            f"States: {s['states']}\nActions: {s['actions']}\n"
            f"Avg V: {s['avg_value']:.4f}\nMax V: {s['max_value']:.4f}\n"
            f"γ = {s['gamma']}\nFeedback: {s['feedback_count']}")
        self.chat.add("System", f"MDP solved ({solver}) — start chatting!", False)
        ToastManager.show(f"MDP solved with {solver}!", "success")
        self._refresh_graph()

    def _ql_callback(self, ep, delta, total):
        self.progress.setValue(ep)
        if self.cb_live.isChecked() and (ep < 5 or ep % 50 == 0):
            self.vi_log.append(f"ep {ep:>4d}  ΣΔQ={delta:.6f}")
        if ep % 10 == 0:
            QApplication.processEvents()

    def _sample(self):
        self.train_text.setPlainText(SAMPLE_TEXT)
        ToastManager.show("Sample text loaded", "info")

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text (*.txt);;All (*)")
        if path:
            with open(path, encoding='utf-8', errors='ignore') as f:
                self.train_text.setPlainText(f.read())

    def _save_model(self):
        if not self.engine.trained:
            ToastManager.show("Train a model first!", "warning"); return
        name, ok = QInputDialog.getText(self, "Save Model", "Model name:", text="MDP_model")
        if ok and name:
            rec = ModelRecord(name, "mdp", self.engine)
            self._saved_models[rec.id] = rec
            path, _ = QFileDialog.getSaveFileName(self, "Save", f"{name}.pkl", "Pickle (*.pkl)")
            if path:
                with open(path, 'wb') as f:
                    pickle.dump(self.engine, f)
            ToastManager.show(f"Model '{name}' saved!", "success")

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Pickle (*.pkl)")
        if path:
            try:
                with open(path, 'rb') as f:
                    self.engine = pickle.load(f)
                s = self.engine.get_stats()
                self.lbl_stats.setText(f"✅ Loaded!\n" + "\n".join(f"{k}: {v}" for k, v in s.items()))
                if self.engine.convergence_history:
                    self.conv_chart.set_data(self.engine.convergence_history)
                self.chat.add("System", f"Model loaded", False)
                self._refresh_graph()
                ToastManager.show("Model loaded!", "success")
            except Exception as e:
                ToastManager.show(f"Load failed: {e}", "error")

    def _open_manager(self):
        if not self._saved_models:
            ToastManager.show("No saved models yet.", "info"); return
        dlg = ModelManagerDialog(self._saved_models, self)
        if dlg.exec() == QDialog.Accepted and dlg.selected_id:
            rec = self._saved_models[dlg.selected_id]
            self.engine = rec.engine
            s = self.engine.get_stats()
            self.lbl_stats.setText(f"✅ Loaded '{rec.name}'!\n" + "\n".join(f"{k}: {v}" for k, v in s.items()))
            self._refresh_graph()

    def _open_explorer(self):
        if not self.engine.trained:
            ToastManager.show("Train a model first!", "warning"); return
        dlg = MDPExplorerDialog(self.engine, self)
        dlg.exec()

    def _on_msg(self, text):
        if not self.engine.trained:
            self.chat.add("Bot", "Train the model first!", False); return
        self._context.append(text)
        self.lbl_context_mdp.setText(f"Context: {len(self._context)} msgs")
        if self.chk_learn_mdp.isChecked():
            self.engine.learn_from_conversation(text)
        self.chat.start_typing()
        QTimer.singleShot(random.randint(200, 600), lambda: self._respond(text))

    def _respond(self, text):
        resp, conf = self.engine.respond(text, context=self._context)
        self.chat.stop_typing_and_respond(resp, conf)
        self._context.append(resp)
        self.lbl_context_mdp.setText(f"Context: {len(self._context)} msgs")

    def _on_feedback(self, user_msg, bot_msg, positive):
        if self.engine.apply_feedback(user_msg, bot_msg, positive):
            self.chat.add("System", "📝 Reward updated from feedback", False)
            ToastManager.show("Feedback applied to MDP rewards", "success")

    def _export_html(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", "chat_mdp.html", "HTML (*.html)")
        if path:
            self.chat.export_html(path)
            ToastManager.show(f"Exported to {os.path.basename(path)}", "success")

    def _export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", "chat_mdp.json", "JSON (*.json)")
        if path:
            self.chat.export_json(path)
            ToastManager.show(f"Exported to {os.path.basename(path)}", "success")

    def _refresh_graph(self):
        if self.engine.trained:
            self.graph.set_data(*self.engine.get_graph_data(25))
            self.val_table.clear()
            self.val_table.append("  SID   Value    Policy→   Text")
            self.val_table.append("  " + "─" * 60)
            top = sorted(self.engine.values.items(), key=lambda x: x[1], reverse=True)[:20]
            for sid, v in top:
                txt = self.engine.states[sid]["text"][:30] if sid in self.engine.states else ""
                pol = f"→ S{self.engine.policy[sid]}" if sid in self.engine.policy else "→ —"
                self.val_table.append(f"  S{sid:<4} {v:>7.3f}  {pol:<8} {txt}")


# ════════════════════════════════════════════════════════════════
#  ANALYSIS TAB
# ════════════════════════════════════════════════════════════════

class AnalysisTab(QWidget):
    def __init__(self, mc_tab: MarkovChainTab, mdp_tab: MDPTab, parent=None):
        super().__init__(parent)
        self.mc_tab = mc_tab
        self.mdp_tab = mdp_tab
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        title = QLabel("📊  Model Analysis")
        title.setStyleSheet(f"color:{ACCENT_ANA};font-size:18px;font-weight:bold;padding:4px;")
        title.setAlignment(Qt.AlignCenter); lay.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)

        # MC side
        left = QWidget(); ll = QVBoxLayout(left)
        lh = QLabel("🔗 Markov Chain")
        lh.setStyleSheet(f"color:{ACCENT_MC};font-size:14px;font-weight:bold;")
        lh.setAlignment(Qt.AlignCenter); ll.addWidget(lh)
        ll.addWidget(QLabel("Word Frequency:"))
        self.mc_freq = WordFreqChart(); ll.addWidget(self.mc_freq, 1)
        ll.addWidget(QLabel("Top Transitions:"))
        self.mc_trans = QTextEdit(); self.mc_trans.setReadOnly(True); self.mc_trans.setMaximumHeight(180)
        self.mc_trans.setStyleSheet(
            f"QTextEdit{{background:{theme()['bg2']};color:{theme()['text']};border:1px solid {theme()['border']};"
            "border-radius:4px;padding:5px;font-family:Consolas,monospace;font-size:10px;}")
        ll.addWidget(self.mc_trans)
        ll.addWidget(styled_btn("🔄 Refresh MC Analysis", ACCENT_MC, self._refresh_mc))
        splitter.addWidget(left)

        # MDP side
        right = QWidget(); rl = QVBoxLayout(right)
        rh = QLabel("🎯 MDP")
        rh.setStyleSheet(f"color:{ACCENT_MDP};font-size:14px;font-weight:bold;")
        rh.setAlignment(Qt.AlignCenter); rl.addWidget(rh)
        rl.addWidget(QLabel("Convergence:"))
        self.mdp_conv = ConvergenceChart(); rl.addWidget(self.mdp_conv, 1)
        rl.addWidget(QLabel("Value Distribution:"))
        self.mdp_vals = QTextEdit(); self.mdp_vals.setReadOnly(True); self.mdp_vals.setMaximumHeight(180)
        self.mdp_vals.setStyleSheet(
            f"QTextEdit{{background:{theme()['bg2']};color:{theme()['text']};border:1px solid {theme()['border']};"
            "border-radius:4px;padding:5px;font-family:Consolas,monospace;font-size:10px;}")
        rl.addWidget(self.mdp_vals)
        rl.addWidget(styled_btn("🔄 Refresh MDP Analysis", ACCENT_MDP, self._refresh_mdp))
        splitter.addWidget(right)

        lay.addWidget(splitter, 1)

    def _refresh_mc(self):
        mc = self.mc_tab.engine
        if mc.trained:
            self.mc_freq.set_data(mc.get_word_frequencies(15))
            trans = mc.get_top_transitions(15)
            self.mc_trans.clear()
            self.mc_trans.append("  From State          → To     Count  Prob")
            self.mc_trans.append("  " + "─" * 55)
            for src, tgt, cnt, prob in trans:
                self.mc_trans.append(f"  {src[:18]:<18} → {tgt:<6}  {cnt:>5}  {prob:.3f}")
        else:
            self.mc_freq.set_data([])
            self.mc_trans.setText("Train a Markov Chain model first.")

    def _refresh_mdp(self):
        mdp = self.mdp_tab.engine
        if mdp.trained:
            if mdp.convergence_history:
                self.mdp_conv.set_data(mdp.convergence_history)
            vals = list(mdp.values.values())
            if vals:
                self.mdp_vals.clear()
                self.mdp_vals.append("Value Distribution:")
                bins = 10
                vmin, vmax = min(vals), max(vals)
                step = (vmax - vmin) / bins if vmax != vmin else 1
                for b in range(bins):
                    lo = vmin + b * step
                    hi = lo + step
                    count = sum(1 for v in vals if lo <= v < hi)
                    bar = "█" * min(count, 50) + f"  ({count})"
                    self.mdp_vals.append(f"  {lo:>6.3f}-{hi:>6.3f}  {bar}")
                self.mdp_vals.append(f"\n  States: {len(vals)}  Mean: {sum(vals)/len(vals):.4f}  Max: {max(vals):.4f}")
        else:
            self.mdp_conv.clear()
            self.mdp_vals.setText("Train an MDP model first.")


# ════════════════════════════════════════════════════════════════
#  COMPARISON TAB
# ════════════════════════════════════════════════════════════════

class CompareTab(QWidget):
    def __init__(self, mc_tab: MarkovChainTab, mdp_tab: MDPTab, parent=None):
        super().__init__(parent)
        self.mc_tab = mc_tab
        self.mdp_tab = mdp_tab
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        title = QLabel("⚖  Side-by-Side Comparison")
        title.setStyleSheet(f"color:{ACCENT_CMP};font-size:18px;font-weight:bold;padding:4px;")
        title.setAlignment(Qt.AlignCenter); lay.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget(); ll = QVBoxLayout(left)
        ll.addWidget(QLabel("🔗 Markov Chain"))
        self.mc_chat = ChatWidget(accent=ACCENT_MC)
        ll.addWidget(self.mc_chat); splitter.addWidget(left)

        right = QWidget(); rl = QVBoxLayout(right)
        rl.addWidget(QLabel("🎯 MDP"))
        self.mdp_chat = ChatWidget(accent=ACCENT_MDP, show_feedback=True)
        self.mdp_chat.feedback_given.connect(self._on_mdp_feedback)
        rl.addWidget(self.mdp_chat); splitter.addWidget(right)
        lay.addWidget(splitter, 1)

        row = QHBoxLayout()
        self.input = QLineEdit(); self.input.setPlaceholderText("Send to both bots…")
        self.input.setStyleSheet(f"""
            QLineEdit{{background:{theme()['input']};color:{theme()['text']};
            border:2px solid {theme()['border']};border-radius:20px;padding:10px 20px;font-size:14px;}}
            QLineEdit:focus{{border-color:{ACCENT_CMP};}}""")
        self.input.returnPressed.connect(self._send)
        row.addWidget(self.input, 1)
        row.addWidget(styled_btn("Send to Both", ACCENT_CMP, self._send))
        lay.addLayout(row)

    def _send(self):
        t = self.input.text().strip()
        if not t: return
        self.mc_chat.add("You", t, True)
        self.mdp_chat.add("You", t, True)
        self.input.clear()

        mc = self.mc_tab.engine
        if mc.trained:
            r, conf = mc.generate(seed=t, max_words=self.mc_tab.sp_max.value(),
                                   temperature=self.mc_tab.sp_temp.value())
            if not r.strip() or r.startswith("⚠"):
                r, conf = mc.generate(max_words=self.mc_tab.sp_max.value())
        else:
            r, conf = "⚠ Not trained", 0
        self.mc_chat.add("MC Bot", r, False)

        mdp = self.mdp_tab.engine
        if mdp.trained:
            r2, conf2 = mdp.respond(t)
        else:
            r2, conf2 = "⚠ Not trained", 0
        self.mdp_chat.add("MDP Bot", r2, False)

    def _on_mdp_feedback(self, user_msg, bot_msg, positive):
        self.mdp_tab.engine.apply_feedback(user_msg, bot_msg, positive)
        self.mdp_chat.add("System", "📝 Reward updated", False)


# ════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Markov Chatbot Studio v3.0")
        self.setMinimumSize(1300, 780)
        self.resize(1540, 880)
        self._apply_theme()

        central = QWidget(); self.setCentralWidget(central)
        ml = QVBoxLayout(central); ml.setContentsMargins(6, 6, 6, 6)

        # Header
        hdr = QLabel("🤖  Markov Chatbot Studio  v3.0")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet(f"""
            color:white; font-size:20px; font-weight:bold; padding:8px;
            background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                {theme()['header_grad']});
            border-radius:10px;""")
        ml.addWidget(hdr)

        # Tabs
        self.tabs = QTabWidget()
        self.mc_tab  = MarkovChainTab()
        self.mdp_tab = MDPTab()
        self.ana_tab = AnalysisTab(self.mc_tab, self.mdp_tab)
        self.cmp_tab = CompareTab(self.mc_tab, self.mdp_tab)

        self.tabs.addTab(self.mc_tab,  "🔗  Markov Chain")
        self.tabs.addTab(self.mdp_tab, "🎯  MDP")
        self.tabs.addTab(self.ana_tab, "📊  Analysis")
        self.tabs.addTab(self.cmp_tab, "⚖  Compare")
        self.tabs.currentChanged.connect(self._tab_changed)
        ml.addWidget(self.tabs, 1)

        self._build_menu()
        self.statusBar().showMessage(
            "Ready — train a model and start chatting!  |  Ctrl+T=Train  Ctrl+S=Save  Ctrl+E=Export  Ctrl+K=Shortcuts")

    def _tab_changed(self, idx):
        if idx == 2:
            self.ana_tab._refresh_mc()
            self.ana_tab._refresh_mdp()

    def _apply_theme(self):
        t = theme()
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{t['bg']}; color:{t['text']}; }}
            QTabWidget::pane {{ border:2px solid {t['border']}; border-radius:8px; background:{t['bg']}; }}
            QTabBar::tab {{ background:{t['bg2']}; color:{t['text']}; padding:12px 28px;
                           border-top-left-radius:8px; border-top-right-radius:8px;
                           font-size:13px; font-weight:bold; margin-right:2px; }}
            QTabBar::tab:selected {{ background:{t['border']}; color:white; }}
            QTabBar::tab:hover {{ background:{t['border']}aa; }}
            QSpinBox, QDoubleSpinBox, QComboBox {{ background:{t['bg2']}; color:{t['text']};
                border:1px solid {t['border']}; border-radius:4px; padding:4px; }}
            QCheckBox {{ color:{t['dim']}; }}
            QCheckBox::indicator {{ width:14px; height:14px; }}
            QCheckBox::indicator:checked {{ background:{t['accent2']}; border-radius:3px; }}
            QCheckBox::indicator:unchecked {{ background:{t['bg2']}; border:1px solid #444; border-radius:3px; }}
            QScrollBar:vertical {{ background:{t['bg2']}; width:8px; border-radius:4px; }}
            QScrollBar::handle:vertical {{ background:{t['accent2']}; border-radius:4px; min-height:30px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
            QStatusBar {{ background:{t['bg2']}; color:{t['dim']}; padding:4px; font-size:11px; }}
            QLabel {{ color:{t['text']}; }}
            QSplitter::handle {{ background:{t['border']}; width:3px; }}
        """)
        pal = QPalette()
        for role, col in [
            (QPalette.Window, QColor(t['bg'])), (QPalette.WindowText, QColor(t['text'])),
            (QPalette.Base, QColor(t['bg2'])), (QPalette.Text, QColor(t['text'])),
            (QPalette.Button, QColor(t['bg2'])), (QPalette.ButtonText, QColor(t['text'])),
            (QPalette.Highlight, QColor(t['accent2'])), (QPalette.HighlightedText, Qt.white),
        ]:
            pal.setColor(role, col)
        QApplication.instance().setPalette(pal)

    def _switch_theme(self, name):
        global CURRENT_THEME
        CURRENT_THEME = name
        self._apply_theme()
        ToastManager.show(f"Theme: {name}", "info")

    def _build_menu(self):
        t = theme()
        mb = self.menuBar()
        mb.setStyleSheet(f"""
            QMenuBar{{background:{t['bg2']};color:{t['text']};padding:4px;}}
            QMenuBar::item:selected{{background:{t['border']};}}
            QMenu{{background:{t['bg2']};color:{t['text']};border:1px solid {t['border']};}}
            QMenu::item:selected{{background:{t['accent2']};}}""")

        fm = mb.addMenu("File")
        self._add_action(fm, "Clear All Chats", "Ctrl+Shift+C", self._clear_all)
        fm.addSeparator()
        self._add_action(fm, "Quit", "Ctrl+Q", self.close)

        am = mb.addMenu("Actions")
        self._add_action(am, "Train Active Model", "Ctrl+T", self._train_active)
        self._add_action(am, "Save Active Model", "Ctrl+S", self._save_active)
        self._add_action(am, "Export Chat HTML", "Ctrl+E", self._export_active)
        self._add_action(am, "Export Chat JSON", "Ctrl+Shift+E", self._export_json_active)

        tm = mb.addMenu("Theme")
        for name in THEMES:
            tm.addAction(name, lambda n=name: self._switch_theme(n))

        hm = mb.addMenu("Help")
        self._add_action(hm, "Keyboard Shortcuts", "Ctrl+K", self._show_shortcuts)
        hm.addAction("About", self._about)

    def _add_action(self, menu, text, shortcut, slot):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        menu.addAction(a)

    def _clear_all(self):
        self.mc_tab.chat.clear_chat()
        self.mdp_tab.chat.clear_chat()
        self.cmp_tab.mc_chat.clear_chat()
        self.cmp_tab.mdp_chat.clear_chat()
        ToastManager.show("All chats cleared", "info")

    def _train_active(self):
        idx = self.tabs.currentIndex()
        if idx == 0: self.mc_tab._train()
        elif idx == 1: self.mdp_tab._train()

    def _save_active(self):
        idx = self.tabs.currentIndex()
        if idx == 0: self.mc_tab._save_model()
        elif idx == 1: self.mdp_tab._save_model()

    def _export_active(self):
        idx = self.tabs.currentIndex()
        if idx == 0: self.mc_tab._export_html()
        elif idx == 1: self.mdp_tab._export_html()

    def _export_json_active(self):
        idx = self.tabs.currentIndex()
        if idx == 0: self.mc_tab._export_json()
        elif idx == 1: self.mdp_tab._export_json()

    def _about(self):
        QMessageBox.about(self, "About",
            "<h2>🤖 Markov Chatbot Studio v3.0</h2>"
            "<p>Build chatbots with Markov Chains & MDPs.</p>"
            "<h3>Key Features</h3><ul>"
            "<li>Markov Chain with n-gram order, temperature, batch generation</li>"
            "<li>MDP with Value Iteration & Q-Learning solvers</li>"
            "<li>Interactive MDP state explorer</li>"
            "<li>Transition probability heatmap</li>"
            "<li>Perplexity scoring</li>"
            "<li>Real-time learning from conversations</li>"
            "<li>Multi-model manager (save, rename, switch)</li>"
            "<li>User feedback → online reward updates</li>"
            "<li>Quick reply suggestions</li>"
            "<li>4 color themes</li>"
            "<li>Toast notifications</li>"
            "<li>Chat search, export (HTML/JSON)</li>"
            "<li>Drag & drop file loading</li>"
            "<li>Auto-save</li>"
            "<li>Force-directed graph layout</li>"
            "<li>Convergence charts</li>"
            "</ul><p><i>Built with PySide6 & Python</i></p>")

    def _show_shortcuts(self):
        QMessageBox.information(self, "Keyboard Shortcuts",
            "<h3>⌨ Shortcuts</h3><table cellpadding='4'>"
            "<tr><td><b>Ctrl+T</b></td><td>Train active model</td></tr>"
            "<tr><td><b>Ctrl+S</b></td><td>Save active model</td></tr>"
            "<tr><td><b>Ctrl+E</b></td><td>Export chat as HTML</td></tr>"
            "<tr><td><b>Ctrl+Shift+E</b></td><td>Export chat as JSON</td></tr>"
            "<tr><td><b>Ctrl+Shift+C</b></td><td>Clear all chats</td></tr>"
            "<tr><td><b>Ctrl+K</b></td><td>Show shortcuts</td></tr>"
            "<tr><td><b>Ctrl+Q</b></td><td>Quit</td></tr>"
            "<tr><td><b>Enter</b></td><td>Send chat message</td></tr>"
            "<tr><td><b>↑ / ↓</b></td><td>Navigate input history</td></tr>"
            "<tr><td><b>Double-click graph</b></td><td>Force-directed layout</td></tr>"
            "<tr><td><b>Scroll on graph</b></td><td>Zoom</td></tr>"
            "<tr><td><b>Right-drag graph</b></td><td>Pan</td></tr>"
            "</table>")


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════

def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()