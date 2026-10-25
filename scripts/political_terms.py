# Two-tier vocabulary for scoring "political discourse" in general Hindi web text.
#
# Why two tiers: the Lok Sabha frequency list is dominated by bureaucratic scheme
# language (योजना, मंत्रालय, ...). Political NEWS/DEBATE text in Sangraha uses a
# different vocabulary (elections, parties, opposition, parliament). Generic admin
# words alone are not enough to call a document political, so:
#   STRONG = clearly political; a doc must contain at least a few DISTINCT ones.
#   WEAK   = policy/administration words; they add to the score but cannot qualify
#            a document on their own.
#
# Matching is exact-token (not substring), so list inflected forms explicitly.
# Deliberately excluded: person names, states/cities, generic sector nouns
# (कृषि, स्वास्थ्य, शिक्षा...), procedural Q&A jargon (माननीय, उपर्युक्त...), and
# ambiguous everyday words (सुरक्षा, सीमा, आधार, बीमा, रोजगार, व्यापार...).

STRONG = {
    # government / office-holders
    "सरकार", "सरकारें", "सरकारों", "प्रधानमंत्री", "मुख्यमंत्री", "मुख्यमंत्रियों",
    "मंत्री", "मंत्रियों", "मंत्रिमंडल", "कैबिनेट", "राष्ट्रपति", "उपराष्ट्रपति",
    "राज्यपाल",
    # legislature
    "संसद", "संसदीय", "लोकसभा", "राज्यसभा", "विधानसभा", "विधानसभाओं",
    "सांसद", "सांसदों", "विधायक", "विधायकों", "सदन", "विधेयक", "अध्यादेश",
    "बहुमत", "अविश्वास",
    # elections / parties / contest
    "चुनाव", "चुनावों", "चुनावी", "निर्वाचन", "मतदान", "मतदाता", "मतदाताओं",
    "प्रत्याशी", "घोषणापत्र", "राजनीति", "राजनीतिक",
    "राजनैतिक", "राजनेता", "राजनेताओं", "राजनीतिज्ञ", "विपक्ष", "विपक्षी",
    "सत्ताधारी", "सत्तापक्ष", "सत्ता", "गठबंधन", "पार्टी", "पार्टियों",
    "भाजपा", "बीजेपी", "कांग्रेस", "वोट",
    "तृणमूल", "शिवसेना", "राजद", "आरजेडी", "जदयू", "सपा", "बसपा", "माकपा", "भाकपा",
    "टीडीपी", "एनडीए", "यूपीए", "प्रतिपक्ष", "ईवीएम", "सियासी", "सियासत",
    "जनप्रतिनिधि", "जनप्रतिनिधियों", "पार्षद", "नगरसेवक",
    # constitution / democracy / judiciary
    "संविधान", "संवैधानिक", "लोकतंत्र", "लोकतांत्रिक", "न्यायपालिका",
}

WEAK = {
    # policy / law
    "योजना", "योजनाओं", "नीति", "नीतियों", "नीतिगत", "अधिनियम", "कानून",
    "कानूनी", "विधि", "संशोधन", "अधिसूचना", "विनियमन", "आरक्षण", "बजट",
    "सब्सिडी", "राजस्व",
    # bodies / actors
    "मंत्रालय", "मंत्रालयों", "आयोग", "प्राधिकरण", "प्रशासन", "प्रशासनिक",
    "अधिकारी", "अधिकारियों", "समिति", "समितियों", "परिषद", "नेता", "नेताओं",
    "दल", "दलों", "अध्यक्ष", "प्रवक्ता", "केंद्रीय", "केन्द्रीय",
    # courts
    "न्यायालय", "न्यायिक", "अदालत", "अदालतों", "सर्वोच्च",
    # contention / accountability
    "आरोप", "आरोपों", "बयान", "आंदोलन", "घोटाला", "भ्रष्टाचार", "विरोध",
    "पारदर्शिता", "जवाबदेही", "उम्मीदवार", "उम्मीदवारों", "रैली", "रैलियों", "सीएम", "पीएम",
}
