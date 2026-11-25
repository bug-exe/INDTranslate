"""
Authentic Indian Sign Language (ISL) Vocabulary & Knowledge Base
Aligned with Indian Sign Language Research and Training Centre (ISLRTC) standards.
Provides semantic metadata, kinematic sign configurations, and reverse sign visualizer data.
"""

from typing import Dict, Any, List, Optional

ISL_SIGNS: Dict[str, Dict[str, Any]] = {
    "namaste": {
        "id": "namaste",
        "label": "Namaste / Hello",
        "label_hi": "नमस्ते / प्रणाम",
        "category": "Greetings",
        "handedness": "Two-handed",
        "description": "Both palms pressed flat together at chest level in prayer gesture (Anjali Mudra), slight head bow.",
        "hand_shape": "Flat open palms, fingers together pointing upward",
        "movement": "Held at chest or slightly raised towards face with forward head nod",
        "facial_expression": "Warm smile, polite eye contact",
        "is_two_handed": True,
        "primary_hand": "both",
        "islrtc_ref": "ISLRTC-GR-001",
        "avatar_frames": [
            {"left_hand": [-0.08, 0.25, 0.1], "right_hand": [0.08, 0.25, 0.1], "head": [0.0, -0.2, 0.05], "pose": "chest_prayer"},
            {"left_hand": [-0.02, 0.20, 0.15], "right_hand": [0.02, 0.20, 0.15], "head": [0.0, -0.15, 0.08], "pose": "chest_prayer_bow"}
        ]
    },
    "thank_you": {
        "id": "thank_you",
        "label": "Thank You",
        "label_hi": "धन्यवाद / शुक्रिया",
        "category": "Politeness",
        "handedness": "One-handed (Dominant)",
        "description": "Fingertips of dominant flat hand touch chin/lips, then extend smoothly forward and slightly upward toward interlocutor.",
        "hand_shape": "Flat B-hand, palm facing chest initially",
        "movement": "Chin outwards toward viewer",
        "facial_expression": "Smiling, appreciative head nod",
        "is_two_handed": False,
        "primary_hand": "right",
        "islrtc_ref": "ISLRTC-PL-004",
        "avatar_frames": [
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.0, 0.0, 0.2], "head": [0.0, -0.1, 0.0], "pose": "touch_chin"},
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.1, 0.15, 0.45], "head": [0.0, -0.05, 0.05], "pose": "extend_forward"}
        ]
    },
    "welcome": {
        "id": "welcome",
        "label": "Welcome",
        "label_hi": "स्वागत है",
        "category": "Greetings",
        "handedness": "Two-handed",
        "description": "Both open hands start outward at side chest level, palms turned up, smoothly sweeping inward towards body center.",
        "hand_shape": "Open palms facing upward",
        "movement": "Lateral to medial inward sweep",
        "facial_expression": "Inviting smile",
        "is_two_handed": True,
        "primary_hand": "both",
        "islrtc_ref": "ISLRTC-GR-008",
        "avatar_frames": [
            {"left_hand": [-0.35, 0.2, 0.3], "right_hand": [0.35, 0.2, 0.3], "head": [0.0, -0.1, 0.0], "pose": "open_arms"},
            {"left_hand": [-0.15, 0.25, 0.2], "right_hand": [0.15, 0.25, 0.2], "head": [0.0, -0.05, 0.0], "pose": "gather_inward"}
        ]
    },
    "please": {
        "id": "please",
        "label": "Please / Request",
        "label_hi": "कृपया",
        "category": "Politeness",
        "handedness": "One-handed",
        "description": "Flat dominant hand placed over center of chest/sternum, making gentle clockwise circular rubbing motions.",
        "hand_shape": "Open flat 5-palm resting against sternum",
        "movement": "Circular rubbing on chest",
        "facial_expression": "Soft, earnest, pleading expression",
        "is_two_handed": False,
        "primary_hand": "right",
        "islrtc_ref": "ISLRTC-PL-002",
        "avatar_frames": [
            {"left_hand": [-0.25, 0.5, 0.0], "right_hand": [0.0, 0.2, 0.1], "head": [0.0, -0.05, 0.0], "pose": "chest_circle_1"},
            {"left_hand": [-0.25, 0.5, 0.0], "right_hand": [0.05, 0.23, 0.1], "head": [0.0, -0.05, 0.0], "pose": "chest_circle_2"}
        ]
    },
    "help": {
        "id": "help",
        "label": "Help / Emergency",
        "label_hi": "मदद / सहायता",
        "category": "Emergency & Needs",
        "handedness": "Two-handed",
        "description": "Non-dominant flat palm held horizontal facing up; dominant hand forms a thumbs-up fist placed on non-dominant palm, both lifted together upward.",
        "hand_shape": "Left flat palm up, Right 'A' fist with thumb extended upward",
        "movement": "Combined upward vertical elevation",
        "facial_expression": "Concerned, urgent gaze",
        "is_two_handed": True,
        "primary_hand": "both",
        "islrtc_ref": "ISLRTC-EM-012",
        "avatar_frames": [
            {"left_hand": [-0.1, 0.35, 0.25], "right_hand": [-0.1, 0.25, 0.25], "head": [0.0, -0.1, 0.0], "pose": "support_base"},
            {"left_hand": [-0.1, 0.20, 0.30], "right_hand": [-0.1, 0.10, 0.30], "head": [0.0, -0.1, 0.0], "pose": "support_lift"}
        ]
    },
    "water": {
        "id": "water",
        "label": "Water",
        "label_hi": "पानी / जल",
        "category": "Daily Needs",
        "handedness": "One-handed",
        "description": "Index, middle, and ring finger extended upright in 'W' shape (or thumb-to-mouth cupping gesture), tapping side of chin/mouth twice.",
        "hand_shape": "W handshape (3 fingers up) or cupped hand",
        "movement": "Double tap against lower lip/chin",
        "facial_expression": "Neutral/anticipatory",
        "is_two_handed": False,
        "primary_hand": "right",
        "islrtc_ref": "ISLRTC-DN-019",
        "avatar_frames": [
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.05, -0.05, 0.2], "head": [0.0, 0.0, 0.0], "pose": "tap_mouth_1"},
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.08, -0.02, 0.25], "head": [0.0, 0.0, 0.0], "pose": "tap_mouth_2"}
        ]
    },
    "food": {
        "id": "food",
        "label": "Food / Hungry",
        "label_hi": "खाना / भोजन / भूख",
        "category": "Daily Needs",
        "handedness": "One-handed",
        "description": "Fingertips of dominant hand clustered together facing upward (flattened O shape), tapping towards mouth repeatedly as if eating morsels.",
        "hand_shape": "Flattened O-hand (bunch of fingertips together)",
        "movement": "Repetitive rhythmic motion to lips",
        "facial_expression": "Slightly open mouth, chewing suggestion",
        "is_two_handed": False,
        "primary_hand": "right",
        "islrtc_ref": "ISLRTC-DN-022",
        "avatar_frames": [
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.02, -0.05, 0.15], "head": [0.0, 0.0, 0.0], "pose": "feed_mouth_in"},
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.04, 0.02, 0.25], "head": [0.0, 0.0, 0.0], "pose": "feed_mouth_out"}
        ]
    },
    "doctor": {
        "id": "doctor",
        "label": "Doctor / Medical",
        "label_hi": "डॉक्टर / चिकित्सक",
        "category": "Medical",
        "handedness": "Two-handed",
        "description": "Non-dominant arm extended forward horizontally with wrist exposed; index and middle finger of dominant hand tap inside of wrist (checking pulse).",
        "hand_shape": "Dominant 'M' or '2' fingers, non-dominant relaxed wrist",
        "movement": "Two gentle taps on radial artery pulse point",
        "facial_expression": "Attentive, professional",
        "is_two_handed": True,
        "primary_hand": "both",
        "islrtc_ref": "ISLRTC-MD-030",
        "avatar_frames": [
            {"left_hand": [-0.15, 0.3, 0.3], "right_hand": [-0.12, 0.28, 0.32], "head": [0.0, -0.05, 0.0], "pose": "pulse_tap_1"},
            {"left_hand": [-0.15, 0.3, 0.3], "right_hand": [-0.10, 0.25, 0.34], "head": [0.0, -0.05, 0.0], "pose": "pulse_tap_2"}
        ]
    },
    "hospital": {
        "id": "hospital",
        "label": "Hospital",
        "label_hi": "अस्पताल",
        "category": "Medical",
        "handedness": "One-handed",
        "description": "Index finger traces the shape of a medical plus cross '+' on the opposite upper arm or shoulder.",
        "hand_shape": "Extended index finger",
        "movement": "Vertical stroke downward followed by horizontal crossing stroke",
        "facial_expression": "Neutral",
        "is_two_handed": False,
        "primary_hand": "right",
        "islrtc_ref": "ISLRTC-MD-033",
        "avatar_frames": [
            {"left_hand": [-0.25, 0.3, 0.1], "right_hand": [-0.2, 0.1, 0.15], "head": [0.0, 0.0, 0.0], "pose": "cross_vert"},
            {"left_hand": [-0.25, 0.3, 0.1], "right_hand": [-0.15, 0.15, 0.15], "head": [0.0, 0.0, 0.0], "pose": "cross_horiz"}
        ]
    },
    "family": {
        "id": "family",
        "label": "Family",
        "label_hi": "परिवार",
        "category": "People & Society",
        "handedness": "Two-handed",
        "description": "Both hands form 'F' handshape (index and thumb touching circle, other fingers fan up), starting with thumbs touching, sweeping outward in a circle until pinkies touch.",
        "hand_shape": "Two 'F' hands",
        "movement": "Outward horizontal circular trajectory",
        "facial_expression": "Warm smile",
        "is_two_handed": True,
        "primary_hand": "both",
        "islrtc_ref": "ISLRTC-PS-041",
        "avatar_frames": [
            {"left_hand": [-0.05, 0.2, 0.3], "right_hand": [0.05, 0.2, 0.3], "head": [0.0, 0.0, 0.0], "pose": "f_touch_start"},
            {"left_hand": [-0.20, 0.2, 0.25], "right_hand": [0.20, 0.2, 0.25], "head": [0.0, 0.0, 0.0], "pose": "f_circle_sweep"},
            {"left_hand": [-0.04, 0.22, 0.2], "right_hand": [0.04, 0.22, 0.2], "head": [0.0, 0.0, 0.0], "pose": "f_touch_end"}
        ]
    },
    "friend": {
        "id": "friend",
        "label": "Friend",
        "label_hi": "दोस्त / मित्र",
        "category": "People & Society",
        "handedness": "Two-handed",
        "description": "Both index fingers hooked together in interlocking grip, alternating position back and forth once.",
        "hand_shape": "Curved hooked index fingers (X-hand)",
        "movement": "Interlock right over left, then left over right",
        "facial_expression": "Friendly grin",
        "is_two_handed": True,
        "primary_hand": "both",
        "islrtc_ref": "ISLRTC-PS-045",
        "avatar_frames": [
            {"left_hand": [-0.06, 0.22, 0.3], "right_hand": [0.02, 0.20, 0.3], "head": [0.0, 0.0, 0.0], "pose": "hook_1"},
            {"left_hand": [-0.02, 0.20, 0.3], "right_hand": [0.06, 0.22, 0.3], "head": [0.0, 0.0, 0.0], "pose": "hook_2"}
        ]
    },
    "deaf": {
        "id": "deaf",
        "label": "Deaf / ISL Signer",
        "label_hi": "बधिर / मूक-बधिर",
        "category": "Community & Identity",
        "handedness": "One-handed",
        "description": "Index finger of dominant hand touches ear, then moves forward to touch corner of mouth.",
        "hand_shape": "Extended index finger",
        "movement": "Ear to mouth arc",
        "facial_expression": "Proud, clear identity",
        "is_two_handed": False,
        "primary_hand": "right",
        "islrtc_ref": "ISLRTC-ID-050",
        "avatar_frames": [
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.18, -0.15, 0.05], "head": [0.0, 0.0, 0.0], "pose": "touch_ear"},
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.08, -0.04, 0.18], "head": [0.0, 0.0, 0.0], "pose": "touch_mouth"}
        ]
    },
    "yes": {
        "id": "yes",
        "label": "Yes / Affirmation",
        "label_hi": "हाँ / सही",
        "category": "Conversation",
        "handedness": "One-handed",
        "description": "Dominant hand forms an 'S' fist held at shoulder height, nodding up and down at wrist like a head nod.",
        "hand_shape": "Closed fist (S-hand)",
        "movement": "Wrist flexion up and down twice",
        "facial_expression": "Nodding head in agreement",
        "is_two_handed": False,
        "primary_hand": "right",
        "islrtc_ref": "ISLRTC-CV-055",
        "avatar_frames": [
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.25, 0.05, 0.35], "head": [0.0, -0.05, 0.05], "pose": "fist_nod_up"},
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.25, 0.15, 0.35], "head": [0.0, 0.05, 0.08], "pose": "fist_nod_down"}
        ]
    },
    "no": {
        "id": "no",
        "label": "No / Negation",
        "label_hi": "नहीं / मना",
        "category": "Conversation",
        "handedness": "One-handed",
        "description": "Index finger extended, wagging left and right rapidly (or index and middle finger closing against thumb like snapping).",
        "hand_shape": "Extended index finger or 3-finger snap",
        "movement": "Rapid lateral side-to-side oscillation",
        "facial_expression": "Head shaking side to side, furrowed brow",
        "is_two_handed": False,
        "primary_hand": "right",
        "islrtc_ref": "ISLRTC-CV-058",
        "avatar_frames": [
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.18, 0.1, 0.35], "head": [-0.08, 0.0, 0.0], "pose": "shake_left"},
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.26, 0.1, 0.35], "head": [0.08, 0.0, 0.0], "pose": "shake_right"}
        ]
    },
    "how_are_you": {
        "id": "how_are_you",
        "label": "How Are You?",
        "label_hi": "आप कैसे हैं?",
        "category": "Questions",
        "handedness": "Two-handed",
        "description": "Both curved hands with knuckles touching chest twist outward smoothly so palms face upward towards interlocutor, paired with questioning head tilt.",
        "hand_shape": "Curved C-hands rotating to open flat palms up",
        "movement": "Twist from chest outwards",
        "facial_expression": "Raised eyebrows, slightly tilted head",
        "is_two_handed": True,
        "primary_hand": "both",
        "islrtc_ref": "ISLRTC-QS-060",
        "avatar_frames": [
            {"left_hand": [-0.08, 0.22, 0.15], "right_hand": [0.08, 0.22, 0.15], "head": [0.0, -0.05, 0.0], "pose": "chest_curved"},
            {"left_hand": [-0.25, 0.25, 0.35], "right_hand": [0.25, 0.25, 0.35], "head": [0.05, -0.08, 0.05], "pose": "palms_up_question"}
        ]
    },
    "good": {
        "id": "good",
        "label": "Good / Fine",
        "label_hi": "अच्छा / ठीक",
        "category": "Evaluation",
        "handedness": "One-handed",
        "description": "Dominant hand forms a clear thumbs-up gesture, or flat hand moves from chin forward into palm of non-dominant hand.",
        "hand_shape": "Thumbs-up (A-hand with extended thumb)",
        "movement": "Firm forward extension with slight upward pop",
        "facial_expression": "Pleasant smile, confident nod",
        "is_two_handed": False,
        "primary_hand": "right",
        "islrtc_ref": "ISLRTC-EV-065",
        "avatar_frames": [
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.2, 0.18, 0.35], "head": [0.0, 0.0, 0.0], "pose": "thumb_pop_1"},
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.2, 0.12, 0.40], "head": [0.0, -0.03, 0.0], "pose": "thumb_pop_2"}
        ]
    },
    "india": {
        "id": "india",
        "label": "India / Bharat",
        "label_hi": "भारत",
        "category": "Geography & Identity",
        "handedness": "One-handed",
        "description": "Dominant hand forms a gentle circular motion above the forehead representing the Ashok Chakra or tricolor wave, finishing with palm open.",
        "hand_shape": "Loose O turning into open waving hand",
        "movement": "Small circular horizontal rotation above head level",
        "facial_expression": "Respectful, proud expression",
        "is_two_handed": False,
        "primary_hand": "right",
        "islrtc_ref": "ISLRTC-NA-070",
        "avatar_frames": [
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.1, -0.25, 0.2], "head": [0.0, 0.0, 0.0], "pose": "chakra_circle_1"},
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.18, -0.22, 0.25], "head": [0.0, 0.0, 0.0], "pose": "chakra_circle_2"}
        ]
    },
    "what": {
        "id": "what",
        "label": "What? (Question)",
        "label_hi": "क्या?",
        "category": "Questions",
        "handedness": "Two-handed",
        "description": "Both open hands held forward at waist/chest level, palms facing up, shaking horizontally back and forth with furrowed brow.",
        "hand_shape": "Flat open hands, palms up",
        "movement": "Horizontal side-to-side flutter",
        "facial_expression": "Furrowed eyebrows, questioning squint",
        "is_two_handed": True,
        "primary_hand": "both",
        "islrtc_ref": "ISLRTC-QS-075",
        "avatar_frames": [
            {"left_hand": [-0.22, 0.3, 0.35], "right_hand": [0.22, 0.3, 0.35], "head": [0.0, -0.05, 0.0], "pose": "shrug_flutter_1"},
            {"left_hand": [-0.26, 0.3, 0.35], "right_hand": [0.26, 0.3, 0.35], "head": [0.0, -0.05, 0.0], "pose": "shrug_flutter_2"}
        ]
    },
    "where": {
        "id": "where",
        "label": "Where? (Location)",
        "label_hi": "कहाँ?",
        "category": "Questions",
        "handedness": "One-handed",
        "description": "Dominant index finger pointed upward, pivoting side to side like a windshield wiper with head tilted in inquiry.",
        "hand_shape": "Extended index finger pointing up",
        "movement": "Side-to-side wrist oscillation",
        "facial_expression": "Searching look, head tilted",
        "is_two_handed": False,
        "primary_hand": "right",
        "islrtc_ref": "ISLRTC-QS-078",
        "avatar_frames": [
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.15, 0.1, 0.35], "head": [0.06, -0.04, 0.0], "pose": "point_sway_left"},
            {"left_hand": [-0.3, 0.5, 0.0], "right_hand": [0.25, 0.1, 0.35], "head": [0.06, -0.04, 0.0], "pose": "point_sway_right"}
        ]
    },
    "love": {
        "id": "love",
        "label": "Love / Affection",
        "label_hi": "प्यार / प्रेम",
        "category": "Feelings",
        "handedness": "Two-handed",
        "description": "Both arms crossed over chest with hands in fists or flat palms hugging shoulders/chest firmly.",
        "hand_shape": "Crossed arms over heart",
        "movement": "Holding chest warmly",
        "facial_expression": "Affectionate gentle smile",
        "is_two_handed": True,
        "primary_hand": "both",
        "islrtc_ref": "ISLRTC-FL-085",
        "avatar_frames": [
            {"left_hand": [0.15, 0.18, 0.1], "right_hand": [-0.15, 0.18, 0.1], "head": [0.0, -0.05, 0.0], "pose": "hug_cross"}
        ]
    }
}

SIGN_INDEX_TO_KEY = list(ISL_SIGNS.keys())
SIGN_KEY_TO_INDEX = {k: i for i, k in enumerate(SIGN_INDEX_TO_KEY)}
TOTAL_SIGN_CLASSES = len(SIGN_INDEX_TO_KEY)

def get_all_signs() -> List[Dict[str, Any]]:
    return list(ISL_SIGNS.values())

def get_sign_by_id(sign_id: str) -> Optional[Dict[str, Any]]:
    return ISL_SIGNS.get(sign_id.lower().strip())

def search_signs(query: str) -> List[Dict[str, Any]]:
    q = query.lower().strip()
    results = []
    for s in ISL_SIGNS.values():
        if q in s["label"].lower() or q in s["label_hi"] or q in s["category"].lower() or q in s["description"].lower():
            results.append(s)
    return results
