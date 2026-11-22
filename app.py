import sys
import random
from collections import defaultdict
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QLabel, QLineEdit, QTextEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QScrollArea, QFrame,
    QDialog, QTableWidget, QTableWidgetItem, QMessageBox, QTabWidget,
    QFileDialog,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor


# ═══════════════════════════════════════════════════════════════════════════════
# MARKOV CHAIN CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
class MarkovChainBot:
    """Generates text by following transition probabilities between word n-grams."""

    def __init__(self, name="MarkovBot", order=2):
        self.name = name
        self.order = order
        self.transitions = defaultdict(list)
        self.starts = []
        self.trained = False

    def train(self, corpus):
        self.transitions.clear()
        self.starts.clear()
        for line in corpus:
            words = line.strip().split()
            if len(words) < self.order + 1:
                continue
            self.starts.append(tuple(words[: self.order]))
            for i in range(len(words) - self.order):
                key = tuple(words[i : i + self.order])
                self.transitions[key].append(words[i + self.order])
        self.trained = len(self.transitions) > 0

    def generate(self, seed_text=None, max_length=30):
        if not self.trained:
            return "I haven't been trained yet. Please provide training data."

        current = None
        if seed_text:
            seed_words = seed_text.split()
            for i in range(max(0, len(seed_words) - self.order), -1, -1):
                key = tuple(seed_words[i : i + self.order])
                if key in self.transitions:
                    current = key
                    break
        if current is None:
            current = random.choice(self.starts) if self.starts else None
        if current is None:
            return "I have nothing to say."

        result = list(current)
        for _ in range(max_length):
            if current in self.transitions and self.transitions[current]:
                next_word = random.choice(self.transitions[current])
                result.append(next_word)
                current = tuple(result[-self.order :])
            else:
                break
        return " ".join(result)

    def get_stats(self):
        return {
            "transitions": sum(len(v) for v in self.transitions.values()),
            "unique_keys": len(self.transitions),
            "start_states": len(self.starts),
            "trained": self.trained,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MDP CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
class MDPBot:
    """Selects responses using a Markov Decision Process with value iteration."""

    STATES = [
        "greeting", "questioning", "informing", "agreeing",
        "disagreeing", "joking", "empathizing", "farewell",
    ]
    ACTIONS = [
        "greet", "ask", "inform", "agree",
        "disagree", "joke", "empathize", "farewell",
    ]

    def __init__(self, name="MDPBot", discount=0.9):
        self.name = name
        self.discount = discount
        self.current_state = "greeting"
        self.transitions = {}
        self.rewards = defaultdict(lambda: defaultdict(float))
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.policy = {}
        self.values = {}
        self.conversation_history = []
        self.trained = False
        self.total_reward = 0.0

        self.response_templates = {
            "greet": [
                "Hello there! How are you doing today?",
                "Hey! Great to chat with you!",
                "Hi! What's on your mind?",
                "Greetings! Wonderful to connect!",
                "Hello! Ready for an interesting conversation.",
            ],
            "ask": [
                "What do you think about that?",
                "Can you tell me more about it?",
                "How does that make you feel?",
                "Why do you think that is the case?",
                "What would you do in that situation?",
            ],
            "inform": [
                "I find that quite interesting to consider.",
                "Based on what I know, that seems reasonable.",
                "That reminds me of an important principle.",
                "The key point here is understanding the context.",
                "From my perspective, that makes a lot of sense.",
            ],
            "agree": [
                "I completely agree with you on that!",
                "You're absolutely right about that.",
                "I couldn't have said it better myself!",
                "That's exactly what I was thinking too!",
                "We are very much on the same page.",
            ],
            "disagree": [
                "I see it differently, actually.",
                "I'm not sure I agree with that perspective.",
                "Interesting view, but I think otherwise.",
                "I have a different take on that matter.",
                "I must respectfully disagree on that point.",
            ],
            "joke": [
                "Why did the neural network go to therapy? Too many issues to process!",
                "Life is like a Markov chain — you never know what state comes next!",
                "I'd tell you a UDP joke, but you might not get it.",
                "Why do programmers prefer dark mode? Light attracts bugs!",
                "There are 10 types of people: those who understand binary and those who don't.",
            ],
            "empathize": [
                "I understand how you feel about that.",
                "That sounds really challenging to deal with.",
                "I appreciate you sharing that with me.",
                "It takes courage to talk about these things.",
                "Your feelings on this are completely valid.",
            ],
            "farewell": [
                "It was great talking to you! Goodbye!",
                "Take care! Until next time.",
                "Farewell! Have a wonderful day ahead!",
                "Goodbye! I enjoyed our conversation.",
                "See you later! Stay well!",
            ],
        }
        self._initialize_defaults()

    def _initialize_defaults(self):
        self.transitions = {
            "greeting": {
                "greet":     {"greeting": 0.2, "questioning": 0.4, "informing": 0.2, "joking": 0.2},
                "ask":       {"questioning": 0.6, "informing": 0.2, "empathizing": 0.2},
                "inform":    {"informing": 0.4, "questioning": 0.3, "agreeing": 0.3},
            },
            "questioning": {
                "ask":       {"questioning": 0.3, "informing": 0.3, "empathizing": 0.2, "joking": 0.2},
                "inform":    {"informing": 0.5, "agreeing": 0.2, "disagreeing": 0.2, "questioning": 0.1},
                "agree":     {"agreeing": 0.4, "informing": 0.3, "questioning": 0.3},
                "joke":      {"joking": 0.4, "questioning": 0.3, "agreeing": 0.3},
            },
            "informing": {
                "inform":    {"informing": 0.3, "questioning": 0.3, "agreeing": 0.2, "disagreeing": 0.2},
                "agree":     {"agreeing": 0.5, "informing": 0.2, "questioning": 0.2, "joking": 0.1},
                "disagree":  {"disagreeing": 0.4, "questioning": 0.3, "informing": 0.3},
                "ask":       {"questioning": 0.5, "empathizing": 0.3, "informing": 0.2},
            },
            "agreeing": {
                "agree":     {"agreeing": 0.3, "informing": 0.3, "joking": 0.2, "questioning": 0.2},
                "inform":    {"informing": 0.4, "questioning": 0.3, "joking": 0.2, "empathizing": 0.1},
                "ask":       {"questioning": 0.5, "informing": 0.2, "agreeing": 0.2, "empathizing": 0.1},
                "joke":      {"joking": 0.4, "agreeing": 0.3, "informing": 0.2, "questioning": 0.1},
            },
            "disagreeing": {
                "disagree":  {"disagreeing": 0.3, "questioning": 0.3, "informing": 0.3, "empathizing": 0.1},
                "inform":    {"informing": 0.4, "questioning": 0.3, "agreeing": 0.2, "disagreeing": 0.1},
                "ask":       {"questioning": 0.5, "empathizing": 0.3, "disagreeing": 0.1, "informing": 0.1},
                "empathize": {"empathizing": 0.4, "agreeing": 0.3, "questioning": 0.2, "informing": 0.1},
            },
            "joking": {
                "joke":      {"joking": 0.3, "agreeing": 0.3, "questioning": 0.2, "informing": 0.2},
                "agree":     {"agreeing": 0.4, "joking": 0.3, "informing": 0.2, "questioning": 0.1},
                "inform":    {"informing": 0.4, "joking": 0.2, "questioning": 0.2, "agreeing": 0.2},
                "ask":       {"questioning": 0.4, "joking": 0.2, "informing": 0.2, "empathizing": 0.2},
            },
            "empathizing": {
                "empathize": {"empathizing": 0.3, "agreeing": 0.2, "informing": 0.2, "questioning": 0.2, "joking": 0.1},
                "agree":     {"agreeing": 0.4, "empathizing": 0.2, "informing": 0.2, "questioning": 0.2},
                "inform":    {"informing": 0.3, "empathizing": 0.2, "questioning": 0.3, "agreeing": 0.2},
                "ask":       {"questioning": 0.4, "empathizing": 0.3, "informing": 0.2, "agreeing": 0.1},
            },
            "farewell": {
                "farewell":  {"greeting": 0.5, "farewell": 0.3, "joking": 0.2},
                "greet":     {"greeting": 0.7, "joking": 0.2, "questioning": 0.1},
            },
        }

        for (s, a), r in {
            ("greeting", "greet"): 5.0, ("greeting", "ask"): 3.0, ("greeting", "inform"): 2.0,
            ("questioning", "ask"): 4.0, ("questioning", "inform"): 3.0,
            ("questioning", "agree"): 3.0, ("questioning", "joke"): 2.0,
            ("informing", "inform"): 3.0, ("informing", "agree"): 4.0,
            ("informing", "disagree"): 2.0, ("informing", "ask"): 3.0,
            ("agreeing", "agree"): 4.0, ("agreeing", "joke"): 3.0,
            ("agreeing", "inform"): 3.0, ("agreeing", "ask"): 2.0,
            ("disagreeing", "ask"): 3.0, ("disagreeing", "empathize"): 4.0,
            ("disagreeing", "disagree"): 2.0, ("disagreeing", "inform"): 3.0,
            ("joking", "joke"): 5.0, ("joking", "agree"): 4.0,
            ("joking", "inform"): 2.0, ("joking", "ask"): 2.0,
            ("empathizing", "empathize"): 5.0, ("empathizing", "agree"): 3.0,
            ("empathizing", "ask"): 3.0, ("empathizing", "inform"): 2.0,
            ("farewell", "farewell"): 5.0, ("farewell", "greet"): 2.0,
        }.items():
            self.rewards[s][a] = r

    def value_iteration(self, iterations=100, threshold=0.001):
        self.values = {s: 0.0 for s in self.STATES}
        for _ in range(iterations):
            new_values = {}
            max_delta = 0
            for state in self.STATES:
                best_value = float("-inf")
                best_action = None
                for action in self.ACTIONS:
                    if state in self.transitions and action in self.transitions[state]:
                        exp_val = self.rewards[state].get(action, 0)
                        for ns, prob in self.transitions[state][action].items():
                            exp_val += self.discount * prob * self.values.get(ns, 0)
                        self.q_values[state][action] = exp_val
                        if exp_val > best_value:
                            best_value = exp_val
                            best_action = action
                if best_action is None:
                    best_value = 0
                    best_action = random.choice(self.ACTIONS)
                new_values[state] = best_value
                self.policy[state] = best_action
                max_delta = max(max_delta, abs(new_values[state] - self.values[state]))
            self.values = new_values
            if max_delta < threshold:
                break
        self.trained = True
        return self.values

    def classify_input(self, text):
        text_lower = text.lower()
        keywords = {
            "greeting":    ["hello", "hi", "hey", "greetings", "howdy", "good morning"],
            "questioning": ["what", "why", "how", "when", "where", "who", "?", "do you", "can you"],
            "agreeing":    ["yes", "agree", "right", "correct", "exactly", "absolutely", "sure", "indeed"],
            "disagreeing": ["no", "disagree", "wrong", "incorrect", "not really", "but", "however"],
            "joking":      ["joke", "funny", "haha", "lol", "laugh", "humor"],
            "empathizing": ["feel", "sad", "happy", "sorry", "understand", "emotion", "difficult"],
            "farewell":    ["bye", "goodbye", "farewell", "see you", "later", "good night"],
        }
        scores = {s: sum(1 for w in ws if w in text_lower) for s, ws in keywords.items()}
        scores["informing"] = 0.5
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "informing"

    def select_action(self, state=None, epsilon=0.1):
        if state is None:
            state = self.current_state
        if not self.trained:
            self.value_iteration()
        if random.random() < epsilon:
            return random.choice(self.ACTIONS)
        return self.policy.get(state, random.choice(self.ACTIONS))

    def transition(self, action):
        if self.current_state in self.transitions and action in self.transitions[self.current_state]:
            t = self.transitions[self.current_state][action]
            self.current_state = random.choices(list(t.keys()), weights=list(t.values()), k=1)[0]
        else:
            self.current_state = random.choice(self.STATES)
        return self.current_state

    def generate_response(self, input_text=None, epsilon=0.1):
        if input_text:
            self.current_state = self.classify_input(input_text)
        action = self.select_action(epsilon=epsilon)
        reward = self.rewards.get(self.current_state, {}).get(action, 0)
        self.total_reward += reward
        old_state = self.current_state
        new_state = self.transition(action)
        response = random.choice(self.response_templates[action])
        self.conversation_history.append(
            {"state": old_state, "action": action, "next_state": new_state, "reward": reward}
        )
        self.current_state = new_state
        return response, action, old_state, new_state, reward

    def update_reward(self, state, action, delta):
        self.rewards[state][action] += delta
        self.trained = False

    def get_transition_graph(self):
        if not self.trained:
            self.value_iteration()
        lines = []
        for state in self.STATES:
            lines.append(f"┌─ {state.upper()} ─────────────────────────")
            actions = self.transitions.get(state, {})
            for action, trans in actions.items():
                reward = self.rewards.get(state, {}).get(action, 0)
                policy_marker = " ★" if self.policy.get(state) == action else ""
                prob_str = ", ".join(f"{s}:{p:.0%}" for s, p in trans.items())
                lines.append(f"│  {action:10s} (R={reward:5.1f}){policy_marker}")
                lines.append(f"│    → {prob_str}")
            lines.append("└─────────────────────────────────────────")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE CORPUS
# ═══════════════════════════════════════════════════════════════════════════════
SAMPLE_CORPUS = [
    "The weather is quite pleasant today and I hope it stays this way for the weekend",
    "I have been thinking about the nature of artificial intelligence and how it evolves",
    "Machine learning algorithms can discover patterns that humans might miss entirely",
    "The beauty of mathematics lies in its ability to describe the universe precisely",
    "Conversations between minds create a tapestry of shared understanding and knowledge",
    "Every day brings new opportunities to learn and grow as individuals and as a society",
    "The philosophy of mind explores the deep questions about consciousness and awareness",
    "Technology has transformed the way we communicate and connect with each other now",
    "In the realm of science fiction many dreams have become reality through innovation",
    "The intersection of creativity and logic produces the most elegant solutions",
    "Understanding probability helps us make better decisions in uncertain situations",
    "Language is a remarkable tool that allows us to share complex thoughts and feelings",
    "The universe is vast and full of mysteries waiting to be discovered and explored",
    "Learning from experience is one of the most fundamental aspects of intelligence",
    "The digital revolution has changed every aspect of our daily lives and work",
    "Curiosity drives progress and leads us to ask important questions about the world",
    "Patterns emerge from chaos when we look carefully enough at the data around us",
    "The future belongs to those who embrace change and adapt to new circumstances",
    "Collaboration and communication are essential for solving complex problems together",
    "The art of conversation requires both speaking and listening with genuine interest",
    "Hello there how are you doing today I hope everything is going well for you",
    "What do you think about the latest developments in technology and science today",
    "I agree that we should always keep an open mind when considering new ideas",
    "That is a fascinating perspective and I would love to explore it further with you",
    "Why do you think people are so drawn to stories and narratives about the future",
    "I understand your concern and I think it is a very valid point to consider",
    "Do you believe that machines will ever truly understand human emotions and feelings",
    "The answer to that question depends on how we define understanding and consciousness",
    "Sometimes the best approach is to step back and look at the bigger picture",
    "I disagree with that assessment because I think there are other factors to consider",
    "That is a really funny way to look at it and I appreciate the humor in it",
    "Tell me more about what you think regarding the implications of this discovery",
    "I feel that we are on the verge of something truly remarkable in this field",
    "Goodbye for now and I look forward to continuing this conversation another time",
    "Good morning and welcome to what promises to be an interesting discussion today",
]


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT BUBBLE WIDGET
# ═══════════════════════════════════════════════════════════════════════════════
class ChatBubble(QFrame):
    def __init__(self, text, sender, bubble_type="system", parent=None):
        super().__init__(parent)
        self.bubble_type = bubble_type
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        sender_label = QLabel(sender)
        sender_label.setFont(QFont("Segoe UI", 8, QFont.Bold))

        msg_label = QLabel(text)
        msg_label.setWordWrap(True)
        msg_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        msg_label.setFont(QFont("Segoe UI", 10))

        styles = {
            "user":   ("color:#333333;", "background-color:#E5E5E5;border:none;border-radius:14px;margin:2px 40px 2px 10px;"),
            "markov": ("color:#555555;", "background-color:#FFFFFF;border:1px solid #E0E0E0;border-radius:14px;margin:2px 10px 2px 40px;"),
            "mdp":    ("color:#555555;", "background-color:#FFFFFF;border:1px solid #E0E0E0;border-radius:14px;margin:2px 10px 2px 40px;"),
            "system": ("color:#888888;", "background-color:transparent;border:none;margin:2px 20px 2px 20px;"),
        }
        s_style, b_style = styles.get(bubble_type, styles["system"])
        sender_label.setStyleSheet(s_style)
        self.setStyleSheet(f"ChatBubble{{{b_style}}}")

        layout.addWidget(sender_label)
        layout.addWidget(msg_label)


# ═══════════════════════════════════════════════════════════════════════════════
# BOT CONFIG PANEL
# ═══════════════════════════════════════════════════════════════════════════════
class BotConfigPanel(QGroupBox):
    def __init__(self, bot_type, bot, parent=None):
        super().__init__(parent)
        self.bot_type = bot_type
        self.bot = bot
        self.setup_ui()

    def setup_ui(self):
        icon = "🔗" if self.bot_type == "Markov Chain" else "🧠"
        self.setTitle(f"{icon} {self.bot.name} ({self.bot_type})")
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 11px; border: 1px solid #D0D0D0; "
            "border-radius: 6px; margin-top: 10px; padding-top: 16px; background-color: #FFFFFF; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #333333; }"
        )
        layout = QFormLayout(self)
        layout.setSpacing(6)

        self.name_edit = QLineEdit(self.bot.name)
        self.name_edit.textChanged.connect(self._update_name)
        layout.addRow("Name:", self.name_edit)

        if self.bot_type == "Markov Chain":
            self.order_spin = QSpinBox()
            self.order_spin.setRange(1, 5)
            self.order_spin.setValue(self.bot.order)
            layout.addRow("Chain Order:", self.order_spin)

            self.max_len_spin = QSpinBox()
            self.max_len_spin.setRange(5, 100)
            self.max_len_spin.setValue(30)
            layout.addRow("Max Length:", self.max_len_spin)
        else:
            self.discount_spin = QDoubleSpinBox()
            self.discount_spin.setRange(0.0, 1.0)
            self.discount_spin.setSingleStep(0.05)
            self.discount_spin.setValue(self.bot.discount)
            self.discount_spin.valueChanged.connect(lambda v: (setattr(self.bot, 'discount', v), setattr(self.bot, 'trained', False)))
            layout.addRow("Discount (γ):", self.discount_spin)

            self.epsilon_spin = QDoubleSpinBox()
            self.epsilon_spin.setRange(0.0, 1.0)
            self.epsilon_spin.setSingleStep(0.05)
            self.epsilon_spin.setValue(0.1)
            layout.addRow("Epsilon (ε):", self.epsilon_spin)

            self.state_label = QLabel(self.bot.current_state)
            self.state_label.setStyleSheet("font-weight:bold;color:#333333;")
            layout.addRow("Current State:", self.state_label)

            self.total_reward_label = QLabel("0.00")
            self.total_reward_label.setStyleSheet("font-weight:bold;color:#333333;")
            layout.addRow("Total Reward:", self.total_reward_label)

            policy_btn = QPushButton("📋 View Policy & Q-Values")
            policy_btn.clicked.connect(self.show_policy)
            layout.addRow("", policy_btn)

            vi_btn = QPushButton("🔄 Run Value Iteration")
            vi_btn.clicked.connect(self.run_value_iteration)
            layout.addRow("", vi_btn)

            heatmap_btn = QPushButton("🗺️ Q-Value Heatmap")
            heatmap_btn.clicked.connect(self.show_policy_heatmap)
            layout.addRow("", heatmap_btn)

            graph_btn = QPushButton("📊 Transition Graph")
            graph_btn.clicked.connect(lambda: QMessageBox.information(
                self, "MDP Transition Graph", self.bot.get_transition_graph()
            ))
            layout.addRow("", graph_btn)

            reset_btn = QPushButton("🔃 Reset State to Greeting")
            reset_btn.clicked.connect(self._reset_state)
            layout.addRow("", reset_btn)

    def _update_name(self, name):
        self.bot.name = name
        icon = "🔗" if self.bot_type == "Markov Chain" else "🧠"
        self.setTitle(f"{icon} {name} ({self.bot_type})")

    def _reset_state(self):
        self.bot.current_state = "greeting"
        self.update_state_display()

    def update_state_display(self):
        if self.bot_type == "MDP":
            self.state_label.setText(self.bot.current_state)
            self.total_reward_label.setText(f"{self.bot.total_reward:.2f}")

    def show_policy(self):
        if not self.bot.trained:
            self.bot.value_iteration()

        dialog = QDialog(self)
        dialog.setWindowTitle(f"MDP Policy & Q-Values — {self.bot.name}")
        dialog.setMinimumSize(700, 500)
        dlg_layout = QVBoxLayout(dialog)
        tabs = QTabWidget()

        # Policy tab
        pt = QTableWidget()
        pt.setColumnCount(4)
        pt.setHorizontalHeaderLabels(["State", "Best Action", "State Value", "Reward"])
        pt.setRowCount(len(MDPBot.STATES))
        for i, s in enumerate(MDPBot.STATES):
            pt.setItem(i, 0, QTableWidgetItem(s))
            ba = self.bot.policy.get(s, "N/A")
            pt.setItem(i, 1, QTableWidgetItem(ba))
            pt.setItem(i, 2, QTableWidgetItem(f"{self.bot.values.get(s, 0):.4f}"))
            pt.setItem(i, 3, QTableWidgetItem(f"{self.bot.rewards.get(s, {}).get(ba, 0):.2f}"))
        pt.horizontalHeader().setStretchLastSection(True)
        pt.resizeColumnsToContents()
        tabs.addTab(pt, "Optimal Policy")

        # Q-values tab
        qt = QTableWidget()
        qt.setColumnCount(len(MDPBot.ACTIONS) + 1)
        qt.setHorizontalHeaderLabels(["State"] + MDPBot.ACTIONS)
        qt.setRowCount(len(MDPBot.STATES))
        for i, s in enumerate(MDPBot.STATES):
            qt.setItem(i, 0, QTableWidgetItem(s))
            for j, a in enumerate(MDPBot.ACTIONS):
                qv = self.bot.q_values.get(s, {}).get(a, 0)
                item = QTableWidgetItem(f"{qv:.3f}")
                if a == self.bot.policy.get(s):
                    item.setBackground(QColor(230, 230, 230)) # Minimalist highlight
                qt.setItem(i, j + 1, item)
        qt.horizontalHeader().setStretchLastSection(True)
        qt.resizeColumnsToContents()
        tabs.addTab(qt, "Q-Values")

        # Transitions tab
        rows = []
        for s in MDPBot.STATES:
            for a, t in self.bot.transitions.get(s, {}).items():
                rows.append((s, a, ", ".join(f"{ns}({p:.1f})" for ns, p in t.items())))
        tt = QTableWidget()
        tt.setColumnCount(3)
        tt.setHorizontalHeaderLabels(["State", "Action", "Transitions"])
        tt.setRowCount(len(rows))
        for i, (s, a, t) in enumerate(rows):
            tt.setItem(i, 0, QTableWidgetItem(s))
            tt.setItem(i, 1, QTableWidgetItem(a))
            tt.setItem(i, 2, QTableWidgetItem(t))
        tt.horizontalHeader().setStretchLastSection(True)
        tt.resizeColumnsToContents()
        tabs.addTab(tt, "Transitions")

        dlg_layout.addWidget(tabs)
        dialog.exec()

    def show_policy_heatmap(self):
        if not self.bot.trained:
            self.bot.value_iteration()

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Q-Value Heatmap — {self.bot.name}")
        dialog.setMinimumSize(650, 450)
        layout = QVBoxLayout(dialog)

        table = QTableWidget()
        table.setColumnCount(len(MDPBot.ACTIONS) + 1)
        table.setHorizontalHeaderLabels(["State"] + MDPBot.ACTIONS)
        table.setRowCount(len(MDPBot.STATES))

        all_vals = []
        for s in MDPBot.STATES:
            for a in MDPBot.ACTIONS:
                all_vals.append(self.bot.q_values.get(s, {}).get(a, 0))
        vmin, vmax = min(all_vals), max(all_vals)

        for i, state in enumerate(MDPBot.STATES):
            table.setItem(i, 0, QTableWidgetItem(state))
            for j, action in enumerate(MDPBot.ACTIONS):
                qv = self.bot.q_values.get(state, {}).get(action, 0)
                item = QTableWidgetItem(f"{qv:.2f}")
                item.setTextAlignment(Qt.AlignCenter)

                if vmax > vmin:
                    ratio = (qv - vmin) / (vmax - vmin)
                else:
                    ratio = 0.5

                # Minimalist grayscale heatmap
                gray = int(255 * (1 - ratio))
                item.setBackground(QColor(gray, gray, gray))
                item.setForeground(QColor(255, 255, 255) if ratio > 0.5 else QColor(0, 0, 0))
                table.setItem(i, j + 1, item)

        table.horizontalHeader().setStretchLastSection(True)
        table.resizeColumnsToContents()
        layout.addWidget(table)

        legend = QLabel("⚫ Low Q-Value  →  ⚪ High Q-Value")
        legend.setAlignment(Qt.AlignCenter)
        legend.setStyleSheet("font-weight:bold;font-size:11px;padding:6px;")
        layout.addWidget(legend)

        dialog.exec()

    def run_value_iteration(self):
        values = self.bot.value_iteration()
        txt = "Value Iteration Converged!\n\nState Values & Policy:\n"
        for s, v in sorted(values.items()):
            txt += f"  {s}: V={v:.4f}  →  π={self.bot.policy.get(s, 'N/A')}\n"
        QMessageBox.information(self, "Value Iteration Complete", txt)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Markov Chain & MDP Chatbot Application")
        self.setMinimumSize(1350, 850)

        self.markov_bot = MarkovChainBot(name="MarkovBot", order=2)
        self.markov_bot.train(SAMPLE_CORPUS)
        self.mdp_bot = MDPBot(name="MDPBot", discount=0.9)
        self.mdp_bot.value_iteration()

        self.bot_chat_active = False
        self.bot_chat_timer = QTimer()
        self.bot_chat_timer.timeout.connect(self.bot_chat_step)
        self.bot_chat_speed = 1500
        self.last_speaker = None
        self.bot_turn_counter = 0
        self.message_counts = {"markov": 0, "mdp": 0, "user": 0}

        self._build_ui()
        self._apply_styles()
        self.add_system_message(
            "Welcome to the Markov Chain & MDP Chatbot Application!\n\n"
            "• Type a message and press Enter to chat with both bots\n"
            "• Use '→ Markov' or '→ MDP' to message one bot only\n"
            "• Click 'Start Bot-to-Bot Chat' to watch them converse\n"
            "• Adjust MDP rewards on the right, then re-run Value Iteration\n"
            "• Edit the corpus and retrain the Markov bot anytime\n"
            "• Use 'Train from Chat History' to make Markov learn from live chat!"
        )

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QHBoxLayout(central)
        main_lay.setSpacing(15)
        main_lay.setContentsMargins(15, 15, 15, 15)

        # ---- LEFT ----
        left = QWidget(); left.setMaximumWidth(310); left.setMinimumWidth(290)
        ll = QVBoxLayout(left)

        self.markov_config = BotConfigPanel("Markov Chain", self.markov_bot)
        ll.addWidget(self.markov_config)
        self.mdp_config = BotConfigPanel("MDP", self.mdp_bot)
        ll.addWidget(self.mdp_config)

        tg = QGroupBox("📚 Training Data (Markov Chain)")
        tl = QVBoxLayout(tg)
        self.corpus_text = QTextEdit(); self.corpus_text.setMaximumHeight(120)
        self.corpus_text.setPlainText("\n".join(SAMPLE_CORPUS))
        tl.addWidget(self.corpus_text)
        
        tb = QPushButton("🏋️ Train / Retrain Markov Bot")
        tb.clicked.connect(self.train_markov_bot); tl.addWidget(tb)
        
        auto_btn = QPushButton("💬 Train from Chat History Too")
        auto_btn.clicked.connect(self.auto_train_from_chat)
        tl.addWidget(auto_btn)
        
        ll.addWidget(tg); ll.addStretch()
        main_lay.addWidget(left)

        # ---- CENTER ----
        center = QWidget(); cl = QVBoxLayout(center)

        hdr = QLabel("💬 Conversation Room")
        hdr.setFont(QFont("Segoe UI", 15, QFont.Bold))
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet("color:#333333;padding:6px;border-bottom:1px solid #E0E0E0; margin-bottom:10px;")
        cl.addWidget(hdr)

        self.chat_scroll = QScrollArea(); self.chat_scroll.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop); self.chat_layout.setSpacing(4)
        self.chat_scroll.setWidget(self.chat_container)
        cl.addWidget(self.chat_scroll, stretch=1)

        ig = QGroupBox("✍️ Your Message"); il = QHBoxLayout(ig)
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message and press Enter…")
        self.input_field.returnPressed.connect(self.send_user_message)
        self.input_field.setFont(QFont("Segoe UI", 11))
        sb = QPushButton("📤 Send"); sb.clicked.connect(self.send_user_message)
        sb.setMinimumWidth(90); sb.setFont(QFont("Segoe UI", 10, QFont.Bold))
        il.addWidget(self.input_field, stretch=1); il.addWidget(sb)
        cl.addWidget(ig)

        ctrl = QHBoxLayout()
        self.bot_chat_btn = QPushButton("🤖 Start Bot-to-Bot Chat")
        self.bot_chat_btn.setCheckable(True)
        self.bot_chat_btn.clicked.connect(self.toggle_bot_chat)
        self.bot_chat_btn.setMinimumHeight(36)
        self.bot_chat_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        ctrl.addWidget(self.bot_chat_btn, stretch=1)

        ctrl.addWidget(QLabel("Speed:"))
        self.speed_spin = QSpinBox(); self.speed_spin.setRange(300, 5000)
        self.speed_spin.setSingleStep(100); self.speed_spin.setValue(1500)
        self.speed_spin.setSuffix(" ms")
        self.speed_spin.valueChanged.connect(self._update_speed)
        ctrl.addWidget(self.speed_spin)

        b1 = QPushButton("→ Markov"); b1.setObjectName("markovBtn")
        b1.clicked.connect(lambda: self.send_to_bot(self.markov_bot))
        b2 = QPushButton("→ MDP"); b2.setObjectName("mdpBtn")
        b2.clicked.connect(lambda: self.send_to_bot(self.mdp_bot))
        b3 = QPushButton("🗑️ Clear"); b3.setObjectName("clearBtn")
        b3.clicked.connect(self.clear_chat)
        b4 = QPushButton("💾 Export"); b4.setObjectName("exportBtn")
        b4.clicked.connect(self.export_conversation)
        
        ctrl.addWidget(b1); ctrl.addWidget(b2); ctrl.addWidget(b3); ctrl.addWidget(b4)
        cl.addLayout(ctrl)
        main_lay.addWidget(center, stretch=1)

        # ---- RIGHT ----
        right = QWidget(); right.setMaximumWidth(310); right.setMinimumWidth(290)
        rl = QVBoxLayout(right)

        sg = QGroupBox("📊 Statistics"); sl = QFormLayout(sg)
        self.lbl_total  = QLabel("0")
        self.lbl_markov = QLabel("0")
        self.lbl_mdp    = QLabel("0")
        self.lbl_user   = QLabel("0")
        self.lbl_exch   = QLabel("0")
        for w in (self.lbl_total, self.lbl_markov, self.lbl_mdp, self.lbl_user, self.lbl_exch):
            w.setStyleSheet("font-weight:bold;")
        sl.addRow("Total Messages:", self.lbl_total)
        sl.addRow("Markov Bot:", self.lbl_markov)
        sl.addRow("MDP Bot:", self.lbl_mdp)
        sl.addRow("User:", self.lbl_user)
        sl.addRow("Bot Exchanges:", self.lbl_exch)
        rl.addWidget(sg)

        mg = QGroupBox("🧠 MDP Live State"); ml = QVBoxLayout(mg)
        self.mdp_st = QLabel("State: greeting")
        self.mdp_st.setStyleSheet("font-weight:bold;color:#333333;font-size:12px;")
        ml.addWidget(self.mdp_st)
        self.mdp_act = QLabel("Last Action: —"); ml.addWidget(self.mdp_act)
        self.mdp_rew = QLabel("Last Reward: —"); ml.addWidget(self.mdp_rew)
        self.mdp_tot = QLabel("Accumulated: 0.00")
        self.mdp_tot.setStyleSheet("color:#333333;font-weight:bold;"); ml.addWidget(self.mdp_tot)
        self.mdp_log = QTextEdit(); self.mdp_log.setReadOnly(True)
        self.mdp_log.setMaximumHeight(150); self.mdp_log.setFont(QFont("Consolas", 8))
        ml.addWidget(self.mdp_log)
        rl.addWidget(mg)

        kg = QGroupBox("🔗 Markov Chain Info"); kl = QFormLayout(kg)
        self.mc_tr = QLabel("0"); self.mc_uk = QLabel("0")
        self.mc_ss = QLabel("0"); self.mc_trained = QLabel("Yes ✅")
        for w in (self.mc_tr, self.mc_uk, self.mc_ss, self.mc_trained):
            w.setStyleSheet("font-weight:bold;")
        kl.addRow("Total Transitions:", self.mc_tr)
        kl.addRow("Unique n-grams:", self.mc_uk)
        kl.addRow("Start States:", self.mc_ss)
        kl.addRow("Trained:", self.mc_trained)
        rl.addWidget(kg); self._update_markov_stats()

        rg = QGroupBox("⭐ MDP Reward Adjustment"); rl2 = QFormLayout(rg)
        self.rw_state = QComboBox(); self.rw_state.addItems(MDPBot.STATES)
        self.rw_action = QComboBox(); self.rw_action.addItems(MDPBot.ACTIONS)
        self.rw_delta = QDoubleSpinBox(); self.rw_delta.setRange(-10, 10)
        self.rw_delta.setSingleStep(0.5); self.rw_delta.setValue(1.0)
        rl2.addRow("State:", self.rw_state)
        rl2.addRow("Action:", self.rw_action)
        rl2.addRow("Delta:", self.rw_delta)
        ab = QPushButton("⚡ Adjust Reward"); ab.clicked.connect(self._adjust_reward)
        rl2.addRow(ab); rl.addWidget(rg)

        lg = QGroupBox("📋 Full Log"); ll2 = QVBoxLayout(lg)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 8)); ll2.addWidget(self.log_text)
        rl.addWidget(lg); rl.addStretch()
        main_lay.addWidget(right)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background: #F5F5F5; }
            QGroupBox {
                font-weight: bold; font-size: 11px; border: 1px solid #D0D0D0; 
                border-radius: 6px; margin-top: 10px; padding-top: 16px; 
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #333333;
            }
            QPushButton {
                background: #333333; color: #FFFFFF; border: none; border-radius: 4px;
                padding: 8px 12px; font-weight: bold; font-size: 10px;
            }
            QPushButton:hover { background: #555555; }
            QPushButton:pressed { background: #1A1A1A; }
            QPushButton#markovBtn { background: #455A64; }
            QPushButton#markovBtn:hover { background: #37474F; }
            QPushButton#mdpBtn { background: #546E7A; }
            QPushButton#mdpBtn:hover { background: #455A64; }
            QPushButton#clearBtn { background: transparent; color: #333333; border: 1px solid #333333; }
            QPushButton#clearBtn:hover { background: #F0F0F0; }
            QPushButton#exportBtn { background: transparent; color: #333333; border: 1px solid #333333; }
            QPushButton#exportBtn:hover { background: #F0F0F0; }
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                border: 1px solid #D0D0D0; border-radius: 4px; padding: 4px; background: #FFFFFF; color: #333333;
            }
            QLabel { color: #333333; }
            QScrollArea { border: 1px solid #D0D0D0; background: #F5F5F5; }
        """)

    # ----------------------------------------------------------- helpers
    def add_message(self, text, sender, btype="system"):
        self.chat_layout.addWidget(ChatBubble(text, sender, btype))
        QTimer.singleShot(50, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()))
        self.log_text.append(f"[{sender}] {text}")

    def add_system_message(self, text):
        lbl = QLabel(f"ℹ️ {text}"); lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color:#888888;font-style:italic;font-size:10px;padding:6px;")
        self.chat_layout.addWidget(lbl)

    def _update_stats(self):
        t = sum(self.message_counts.values())
        self.lbl_total.setText(str(t))
        self.lbl_markov.setText(str(self.message_counts["markov"]))
        self.lbl_mdp.setText(str(self.message_counts["mdp"]))
        self.lbl_user.setText(str(self.message_counts["user"]))
        self.lbl_exch.setText(str(self.bot_turn_counter))

    def _update_mdp_info(self, action, old_st, new_st, reward):
        self.mdp_st.setText(f"State: {new_st}")
        self.mdp_act.setText(f"Last Action: {action}")
        self.mdp_rew.setText(f"Last Reward: {reward:.2f}")
        self.mdp_tot.setText(f"Accumulated: {self.mdp_bot.total_reward:.2f}")
        log_entry = f"{old_st} ─[{action}]→ {new_st}  R={reward:.1f}"
        self.mdp_log.append(log_entry)
        if self.mdp_log.document().blockCount() > 200:
            cursor = self.mdp_log.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 50)
            cursor.removeSelectedText()

    def _update_markov_stats(self):
        s = self.markov_bot.get_stats()
        self.mc_tr.setText(str(s["transitions"]))
        self.mc_uk.setText(str(s["unique_keys"]))
        self.mc_ss.setText(str(s["start_states"]))
        self.mc_trained.setText("Yes ✅" if s["trained"] else "No ❌")

    def _update_speed(self, val):
        self.bot_chat_speed = val
        if self.bot_chat_timer.isActive():
            self.bot_chat_timer.setInterval(val)

    # ----------------------------------------------------------- user actions
    def send_user_message(self):
        text = self.input_field.text().strip()
        if not text:
            return
        self.add_message(text, "You", "user")
        self.message_counts["user"] += 1
        self.input_field.clear()

        mk_resp = self.markov_bot.generate(
            seed_text=text, max_length=self.markov_config.max_len_spin.value(),
        )
        self.add_message(mk_resp, self.markov_bot.name, "markov")
        self.message_counts["markov"] += 1

        mdp_resp, action, old_st, new_st, reward = self.mdp_bot.generate_response(
            input_text=text, epsilon=self.mdp_config.epsilon_spin.value()
        )
        self.add_message(mdp_resp, self.mdp_bot.name, "mdp")
        self.message_counts["mdp"] += 1

        self._update_mdp_info(action, old_st, new_st, reward)
        self._update_markov_stats()
        self._update_stats()
        self.mdp_config.update_state_display()

    def send_to_bot(self, bot):
        text = self.input_field.text().strip()
        if not text:
            return
        self.add_message(text, "You", "user")
        self.message_counts["user"] += 1
        self.input_field.clear()

        if bot is self.markov_bot:
            resp = self.markov_bot.generate(
                seed_text=text, max_length=self.markov_config.max_len_spin.value(),
            )
            self.add_message(resp, self.markov_bot.name, "markov")
            self.message_counts["markov"] += 1
        else:
            resp, action, old_st, new_st, reward = self.mdp_bot.generate_response(
                input_text=text, epsilon=self.mdp_config.epsilon_spin.value()
            )
            self.add_message(resp, self.mdp_bot.name, "mdp")
            self.message_counts["mdp"] += 1
            self._update_mdp_info(action, old_st, new_st, reward)
            self.mdp_config.update_state_display()

        self._update_markov_stats()
        self._update_stats()

    # ----------------------------------------------------------- bot-to-bot
    def toggle_bot_chat(self, checked):
        if checked:
            self.bot_chat_active = True
            self.bot_chat_btn.setText("🛑 Stop Bot-to-Bot Chat")
            self.bot_chat_btn.setStyleSheet(
                "background-color:#C62828;color:white;border-radius:4px;"
                "padding:8px 12px;font-weight:bold;font-size:10px;"
            )
            self.add_system_message("🤖 Bot-to-bot conversation started!")
            self.bot_chat_timer.start(self.bot_chat_speed)
        else:
            self.bot_chat_active = False
            self.bot_chat_btn.setText("🤖 Start Bot-to-Bot Chat")
            self.bot_chat_btn.setStyleSheet("") # Revert to main stylesheet
            self.add_system_message("Bot-to-bot conversation stopped.")
            self.bot_chat_timer.stop()

    def bot_chat_step(self):
        if not self.bot_chat_active:
            return

        self.bot_turn_counter += 1

        if self.last_speaker is not self.markov_bot:
            mk_resp = self.markov_bot.generate(
                max_length=self.markov_config.max_len_spin.value()
            )
            self.add_message(mk_resp, self.markov_bot.name, "markov")
            self.message_counts["markov"] += 1

            mdp_resp, action, old_st, new_st, reward = self.mdp_bot.generate_response(
                input_text=mk_resp, epsilon=self.mdp_config.epsilon_spin.value()
            )
            self.add_message(mdp_resp, self.mdp_bot.name, "mdp")
            self.message_counts["mdp"] += 1
            self.last_speaker = self.mdp_bot
            self._update_mdp_info(action, old_st, new_st, reward)
        else:
            mdp_resp, action, old_st, new_st, reward = self.mdp_bot.generate_response(
                epsilon=self.mdp_config.epsilon_spin.value()
            )
            self.add_message(mdp_resp, self.mdp_bot.name, "mdp")
            self.message_counts["mdp"] += 1

            mk_resp = self.markov_bot.generate(
                seed_text=mdp_resp, max_length=self.markov_config.max_len_spin.value(),
            )
            self.add_message(mk_resp, self.markov_bot.name, "markov")
            self.message_counts["markov"] += 1
            self.last_speaker = self.markov_bot
            self._update_mdp_info(action, old_st, new_st, reward)

        self._update_stats()
        self._update_markov_stats()
        self.mdp_config.update_state_display()

    # ----------------------------------------------------------- training, rewards, export
    def train_markov_bot(self):
        corpus = [l.strip() for l in self.corpus_text.toPlainText().split("\n") if l.strip()]
        if not corpus:
            QMessageBox.warning(self, "No Data", "Please enter some training text.")
            return
        self.markov_bot.order = self.markov_config.order_spin.value()
        self.markov_bot.train(corpus)
        self._update_markov_stats()
        self.add_system_message(
            f"Markov Bot retrained with {len(corpus)} lines (order={self.markov_bot.order})."
        )

    def auto_train_from_chat(self):
        existing = [l.strip() for l in self.corpus_text.toPlainText().split("\n") if l.strip()]
        for i in range(self.chat_layout.count()):
            widget = self.chat_layout.itemAt(i).widget()
            if isinstance(widget, ChatBubble):
                msg_label = widget.layout().itemAt(1).widget()
                if msg_label:
                    existing.append(msg_label.text())
        self.markov_bot.order = self.markov_config.order_spin.value()
        self.markov_bot.train(existing)
        self._update_markov_stats()
        self.add_system_message(
            f"Markov Bot retrained with {len(existing)} total lines (including chat history)."
        )

    def _adjust_reward(self):
        state = self.rw_state.currentText()
        action = self.rw_action.currentText()
        delta = self.rw_delta.value()
        self.mdp_bot.update_reward(state, action, delta)
        self.add_system_message(
            f"Reward adjusted: R({state}, {action}) += {delta:.1f}. "
            f"New value: {self.mdp_bot.rewards[state][action]:.2f}. "
            f"Run Value Iteration to update the policy."
        )

    def export_conversation(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Conversation", "conversation.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.log_text.toPlainText())
            self.add_system_message(f"Conversation exported to {path}")

    # ----------------------------------------------------------- clear
    def clear_chat(self):
        while self.chat_layout.count():
            child = self.chat_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.log_text.clear()
        self.mdp_log.clear()
        self.message_counts = {"markov": 0, "mdp": 0, "user": 0}
        self.bot_turn_counter = 0
        self.last_speaker = None
        self.mdp_bot.current_state = "greeting"
        self.mdp_bot.total_reward = 0.0
        self.mdp_config.update_state_display()
        self.mdp_st.setText("State: greeting")
        self.mdp_act.setText("Last Action: —")
        self.mdp_rew.setText("Last Reward: —")
        self.mdp_tot.setText("Accumulated: 0.00")
        self._update_stats()
        self.add_system_message("Chat cleared. Start a new conversation!")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()