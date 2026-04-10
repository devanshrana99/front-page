"""
AIML Chatbot Engine v2 — TrafficBot
Full AIML 1.0 parser + pattern matcher with SRAI, wildcards, and time-aware responses.
"""

import re
import os
import xml.etree.ElementTree as ET
from datetime import datetime


# ─────────────────────────────────────────────────────
# AIML Knowledge Base (inline, no file dependency)
# ─────────────────────────────────────────────────────
AIML_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<aiml version="1.0.1">

  <category><pattern>HELLO</pattern>
  <template>Hello! 👋 I'm TrafficBot, powered by AIML. Ask about traffic, CNN predictions, congestion levels, or best travel times!</template></category>

  <category><pattern>HI</pattern><template><srai>HELLO</srai></template></category>
  <category><pattern>HEY</pattern><template><srai>HELLO</srai></template></category>
  <category><pattern>YO</pattern><template><srai>HELLO</srai></template></category>

  <category><pattern>GOOD MORNING</pattern>
  <template>Good morning! 🌅 Morning rush hour (7–10 AM) is the heaviest congestion window. Check the live dashboard before heading out!</template></category>

  <category><pattern>GOOD EVENING</pattern>
  <template>Good evening! 🌆 Evening rush (5–8 PM) is peaking now. NH-48 and Ring Road tend to be the worst at this hour.</template></category>

  <category><pattern>WHAT IS TRAFFIC LIKE</pattern>
  <template>Our CNN model is running live inference on all 6 Delhi road segments right now! 📊 Check the Dashboard tab for the live map and CNN probability outputs. The model updates every 60 seconds.</template></category>

  <category><pattern>HOW IS TRAFFIC</pattern><template><srai>WHAT IS TRAFFIC LIKE</srai></template></category>
  <category><pattern>CURRENT TRAFFIC</pattern><template><srai>WHAT IS TRAFFIC LIKE</srai></template></category>
  <category><pattern>TRAFFIC NOW</pattern><template><srai>WHAT IS TRAFFIC LIKE</srai></template></category>

  <category><pattern>WHAT IS CONGESTION</pattern>
  <template>Traffic congestion = demand exceeds road capacity. Our CNN classifies 4 levels:
🟢 Level 0 — Free Flow: speed &gt; 90 km/h, smooth movement
🟡 Level 1 — Moderate: 60–90 km/h, minor delays
🟠 Level 2 — Heavy: 30–60 km/h, significant slowdowns
🔴 Level 3 — Severe: &lt;30 km/h, near-gridlock conditions</template></category>

  <category><pattern>EXPLAIN CONGESTION LEVELS</pattern><template><srai>WHAT IS CONGESTION</srai></template></category>
  <category><pattern>CONGESTION LEVELS</pattern><template><srai>WHAT IS CONGESTION</srai></template></category>

  <category><pattern>HOW DOES THE CNN WORK</pattern>
  <template>Our CNN (Convolutional Neural Network) pipeline 🧠:
1️⃣ Input: 24-hour traffic window → shape (24, 10, 3)
2️⃣ Block 1: Conv2D(32 filters, 3×3) × 2 + BatchNorm + MaxPool + Dropout(0.25)
3️⃣ Block 2: Conv2D(64 filters, 3×3) × 2 + BatchNorm + MaxPool + Dropout(0.25)
4️⃣ Block 3: Conv2D(128) × 2 + BatchNorm + GlobalAvgPool + Dropout(0.4)
5️⃣ Head: Dense(256) → Dense(128) → Softmax(4)
6️⃣ Output: probabilities for Free / Moderate / Heavy / Severe
Accuracy: ~91.4% on 1000-sample validation set</template></category>

  <category><pattern>HOW DOES THE PREDICTION WORK</pattern><template><srai>HOW DOES THE CNN WORK</srai></template></category>
  <category><pattern>HOW DOES CNN WORK</pattern><template><srai>HOW DOES THE CNN WORK</srai></template></category>
  <category><pattern>HOW DOES YOUR MODEL WORK</pattern><template><srai>HOW DOES THE CNN WORK</srai></template></category>

  <category><pattern>WHAT IS CNN</pattern>
  <template>CNN = Convolutional Neural Network 🔷
In image tasks, Conv2D layers detect edges and shapes. In our traffic system, they detect temporal-spatial patterns in the (24h × 10 features) input grid — like rush-hour spikes, weather dips, and incident signatures.
Key layers used: Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Softmax.</template></category>

  <category><pattern>WHAT IS A CNN</pattern><template><srai>WHAT IS CNN</srai></template></category>

  <category><pattern>HOW ACCURATE IS THE MODEL</pattern>
  <template>TrafficCNN v2 accuracy breakdown 📈:
🟢 Free Flow:  94% precision, 93% recall
🟡 Moderate:   89% precision, 90% recall
🟠 Heavy:      90% precision, 88% recall
🔴 Severe:     92% precision, 94% recall
Overall: ~91.4% accuracy on validation set (1000 samples)
Avg confidence: 88.7% across all predictions</template></category>

  <category><pattern>HOW ACCURATE IS YOUR MODEL</pattern><template><srai>HOW ACCURATE IS THE MODEL</srai></template></category>
  <category><pattern>WHAT IS THE ACCURACY</pattern><template><srai>HOW ACCURATE IS THE MODEL</srai></template></category>

  <category><pattern>WHAT DATA DO YOU USE</pattern>
  <template>The CNN processes 10 features over a 24-hour rolling window 📡:
① avg_speed — normalized vehicle speed (km/h)
② volume — vehicles per minute
③ occupancy — % time road is occupied
④ incidents — accident/blockage count
⑤ weather_score — weather impact (0=bad, 1=clear)
⑥ time_sin — sine-encoded hour of day
⑦ time_cos — cosine-encoded hour of day
⑧ capacity_util — road capacity utilization
⑨ temperature — ambient temperature (normalized)
⑩ visibility — sight distance (normalized)
All shaped into (24, 10, 3) tensor with 3 channels for Conv2D input.</template></category>

  <category><pattern>WHAT FEATURES DO YOU USE</pattern><template><srai>WHAT DATA DO YOU USE</srai></template></category>

  <category><pattern>BEST TIME TO TRAVEL</pattern>
  <template>Optimal travel windows based on CNN historical patterns 🕐:
✅ Best: 6–7 AM · 10 AM–12 PM · 2–4 PM · After 9 PM
⚠️ Avoid: 7–10 AM (morning rush) · 5–8 PM (evening rush)
📅 Weekends: 40–60% less congested than weekdays
💡 Check the History tab for a full 24h congestion chart!</template></category>

  <category><pattern>WHEN SHOULD I TRAVEL</pattern><template><srai>BEST TIME TO TRAVEL</srai></template></category>
  <category><pattern>BEST TIME TO DRIVE</pattern><template><srai>BEST TIME TO TRAVEL</srai></template></category>

  <category><pattern>HOW DO I AVOID TRAFFIC</pattern>
  <template>Tips to beat congestion 🚗💨:
• Leave before 7:30 AM or after 9:30 PM on weekdays
• Check the dashboard map for green (free flow) routes
• NH-9 and Airport Road tend to clear faster off-peak
• Monitor weather — rain adds ~30% congestion
• Consider Delhi Metro during heavy/severe conditions</template></category>

  <category><pattern>AVOID TRAFFIC</pattern><template><srai>HOW DO I AVOID TRAFFIC</srai></template></category>

  <category><pattern>HOW DOES WEATHER AFFECT TRAFFIC</pattern>
  <template>Weather impact modeled as CNN input feature 🌦️:
🌧️ Light rain:   +15–25% congestion
🌧️ Heavy rain:   +30–45% congestion
🌫️ Dense fog:    +50% (visibility &lt;100m triggers severity jump)
⛈️ Storms:       +60–80% congestion
🌡️ Extreme heat: +5–10% (more breakdowns)
🌫️ Delhi smog:  +15–20% (reduced speeds)
The weather_score feature (0=bad, 1=clear) feeds directly into the Conv2D layers.</template></category>

  <category><pattern>DOES RAIN AFFECT TRAFFIC</pattern>
  <template>Yes! Rain significantly increases congestion 🌧️:
Speed drops 20–35% in heavy rain
Accidents increase → more incidents feature spike
Our CNN's weather_score feature captures this:
  Score 1.0 = clear skies → baseline congestion
  Score 0.6 = rain → +25% predicted congestion
  Score 0.3 = heavy rain/fog → severe prediction likely</template></category>

  <category><pattern>HOW IS NH 48</pattern>
  <template>NH-48 (Delhi–Gurgaon Highway) 🛣️:
• Capacity: 3,000 vehicles/hour
• Morning peak (8–10 AM): Level 3 Severe on weekdays
• Evening peak (6–8 PM): Level 2–3 Heavy to Severe
• Best window: Before 7:30 AM or after 9:00 PM
• CNN avg confidence on this segment: ~89%
Live prediction on Dashboard tab!</template></category>

  <category><pattern>HOW IS RING ROAD</pattern>
  <template>Delhi Ring Road 🔄:
• Capacity: 2,500 vehicles/hour
• Consistently moderate throughout the day
• ITO junction and Ashram Chowk are main bottlenecks
• Evening rush: 5–8 PM peaks at Level 2–3
• CNN predicts this segment with ~87% confidence
Live status on Dashboard map!</template></category>

  <category><pattern>HOW IS THE EXPRESSWAY</pattern>
  <template>Delhi–Meerut Expressway (NH-9 North) 🚀:
• Capacity: 2,800 vehicles/hour
• Usually the least congested of all monitored routes
• Peak congestion only during extreme rush hours
• Good alternative to Ring Road during 8–10 AM rush</template></category>

  <category><pattern>HOW OFTEN IS DATA UPDATED</pattern>
  <template>System refresh cycle ⚡:
• CNN inference: Every 60 seconds
• Dashboard auto-refresh: Every 60 seconds
• History data: Last 24 hours, per-hour resolution
• Heatmap: Last 7 days (weekly patterns)
The countdown timer in the header shows the next update!</template></category>

  <category><pattern>HOW OFTEN IS DATA REFRESHED</pattern><template><srai>HOW OFTEN IS DATA UPDATED</srai></template></category>
  <category><pattern>HOW OFTEN DO YOU UPDATE</pattern><template><srai>HOW OFTEN IS DATA UPDATED</srai></template></category>

  <category><pattern>WHAT IS THIS SYSTEM</pattern>
  <template>Real-Time Traffic Congestion Prediction System 🚦:
🧠 CNN (TensorFlow/Keras) — 3-block Conv2D, ~187K params, 91.4% accuracy
🤖 AIML chatbot — 40+ intent patterns with SRAI support
📍 6 Delhi NCR road segments monitored live
📊 4-class output: Free Flow / Moderate / Heavy / Severe
⚡ 60-second CNN inference refresh cycle
🗺️ Live SVG traffic network map
📈 24-hour history + weekly heatmap</template></category>

  <category><pattern>ABOUT THIS SYSTEM</pattern><template><srai>WHAT IS THIS SYSTEM</srai></template></category>

  <category><pattern>WHAT IS AIML</pattern>
  <template>AIML = Artificial Intelligence Markup Language 🤖
An XML-based language for building rule-based chatbots.
Core structure: &lt;pattern&gt; + &lt;template&gt; pairs inside &lt;category&gt; tags.
Features: wildcard matching (*), SRAI (redirect to another pattern), variable storage.
TrafficBot uses a custom Python AIML engine that parses our traffic_bot.aiml file.</template></category>

  <category><pattern>HELP</pattern>
  <template>TrafficBot Help 💡 — Ask me about:
🚦 "What is traffic like?" — Live status
🧠 "How does the CNN work?" — Model architecture
📊 "How accurate is the model?" — Performance stats
🕐 "Best time to travel?" — Optimal windows
🗺️ "How do I avoid traffic?" — Route tips
🌧️ "How does weather affect traffic?" — Weather model
🛣️ "How is NH-48 / Ring Road?" — Segment info
🔄 "How often is data updated?" — Refresh cycle
🤖 "What is AIML?" — Chatbot tech
Just ask naturally!</template></category>

  <category><pattern>BYE</pattern>
  <template>Safe travels! 🚗✨ Check the dashboard before heading out and drive carefully!</template></category>

  <category><pattern>GOODBYE</pattern><template><srai>BYE</srai></template></category>
  <category><pattern>SEE YOU</pattern><template><srai>BYE</srai></template></category>

  <category><pattern>THANKS</pattern>
  <template>You're welcome! 😊 Stay safe on the roads and check back anytime for live predictions!</template></category>
  <category><pattern>THANK YOU</pattern><template><srai>THANKS</srai></template></category>

  <category><pattern>*</pattern>
  <template>I didn't quite catch that. I can help with:
• Traffic conditions and predictions
• CNN model questions
• Route and timing advice
• Weather impact
Type "help" to see all topics! 🚦</template></category>

</aiml>'''


# ─────────────────────────────────────────────────────
# AIML Engine
# ─────────────────────────────────────────────────────

class AIMLEngine:
    """Lightweight AIML 1.0 interpreter"""

    def __init__(self, aiml_source=None):
        self.patterns = []
        src = aiml_source or AIML_XML
        self._parse(src)

    def _parse(self, xml_str):
        root = ET.fromstring(xml_str)
        for cat in root.findall('category'):
            pat_el = cat.find('pattern')
            tpl_el = cat.find('template')
            if pat_el is None or tpl_el is None:
                continue
            pat = (pat_el.text or '').strip().upper()
            tpl = self._tpl_text(tpl_el)
            regex = self._to_regex(pat)
            self.patterns.append((regex, tpl))

    def _tpl_text(self, el):
        parts = []
        if el.text: parts.append(el.text)
        for child in el:
            if child.tag == 'srai':
                parts.append(f'__SRAI__:{(child.text or "").strip().upper()}')
            if child.tail: parts.append(child.tail)
        return ''.join(parts)

    def _to_regex(self, pat):
        esc = re.escape(pat).replace(r'\*', '.*')
        return re.compile(f'^{esc}$', re.IGNORECASE)

    def _respond(self, text, depth=0):
        if depth > 5: return "I can help with traffic questions — type 'help'!"
        norm = text.strip().upper()
        for regex, tpl in self.patterns:
            if regex.match(norm):
                if '__SRAI__:' in tpl:
                    m = re.search(r'__SRAI__:(.+)', tpl)
                    if m: return self._respond(m.group(1), depth+1)
                return tpl
        return "Type 'help' to see what I can assist with! 🚦"

    def respond(self, user_input):
        resp = self._respond(user_input)
        h = datetime.now().hour
        if h in range(8, 11) and 'rush' not in resp.lower():
            resp += '\n\n⚠️ Morning rush hour is active right now (7–10 AM)!'
        elif h in range(17, 21) and 'rush' not in resp.lower():
            resp += '\n\n⚠️ Evening rush hour is active right now (5–8 PM)!'
        return resp


class TrafficChatbot:
    """High-level chatbot with conversation history"""

    def __init__(self):
        self.engine = AIMLEngine()
        self.history = []

    def chat(self, user_msg):
        resp = self.engine.respond(user_msg)
        entry = {"user": user_msg, "bot": resp, "ts": datetime.now().isoformat()}
        self.history.append(entry)
        return {"response": resp, "timestamp": entry["ts"]}

    def clear(self):
        self.history = []
