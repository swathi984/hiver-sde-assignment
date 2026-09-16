import csv

OUTPUT_PATH = "data/golden/human_llm_agreement.csv"

rows = [
    [
        184,
        "@AmazonHelp thank you",
        "Thanks again for taking the time to report it. This will help us improve our services.",
        3,
        True,
        2,
        True,
        "The reply incorrectly assumes the customer reported an issue when they only said thank you."
    ],
    [
        154,
        "@AmazonHelp Your Customer team did not provide any proper solution when i contacted them.",
        "You will receive an email correspondence from our team here: https://t.co/HObeYwFNxn\nDo keep a check and you may reply to it for any further assistance.",
        3,
        False,
        4,
        False,
        "Provides a clear follow-up through email and is relevant to the complaint, although the wording could be more natural."
    ],
    [
        171,
        "@AmazonHelp ありがとうございます！ お世話になってます。笑",
        "嬉しいお言葉、ありがとうございます♪(*´∇｀*) どうかこれからもhttps://t.co/st4oU5QbhPをよろしくお願いいたします～♪ (*^^*)",
        5,
        False,
        5,
        False,
        "Friendly and relevant response to a positive customer message with an appropriate casual tone."
    ],
    [
        146,
        "@AmazonHelp We already have and we requested for our replacement phone to be sent via a different courier and again you're sending it by @116324 😡",
        "Thanks for confirming. As you've shared your details, we'll reach out to you soon.",
        2,
        True,
        2,
        True,
        "The response does not address the courier problem and incorrectly assumes the customer shared details."
    ],
    [
        145,
        "@AmazonHelp This gets fixed, or I'm gone. I'll sell off my kindle's, all 5 echo devices, etc. Avg $10K a year in purchases. Disgusting treatment.",
        "Thank you for letting us know. I'm glad it's fixed.",
        1,
        True,
        1,
        True,
        "The reply incorrectly assumes the issue is already fixed and fails to address the customer's complaint."
    ]
]

header = [
    "id",
    "customer_message",
    "reply",
    "llm_overall",
    "llm_unsupported_claim",
    "human_overall",
    "human_unsupported_claim",
    "human_notes"
]

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("=" * 70)
print("HUMAN REVIEW CSV FIXED")
print("=" * 70)
print(f"Rows written: {len(rows)}")
print(f"Saved: {OUTPUT_PATH}")