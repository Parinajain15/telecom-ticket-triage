import random
import pandas as pd

# Realistic telecom complaint templates by department
DATA_TEMPLATES = {
    "Network": [
        ("No mobile signal in my area since morning, emergency calls only.", "High"),
        ("5G internet speed drops completely every evening in Bhubaneswar.", "Normal"),
        ("Frequent call drops during important office meetings.", "High"),
        ("Zero network bars inside my apartment for the last 2 days.", "High"),
        ("Mobile data keeps disconnecting while switching towers.", "Normal"),
    ],
    "Billing": [
        ("I was charged twice for my monthly postpaid bill this cycle.", "High"),
        ("Need a copy of my GST invoice for last month's recharge.", "Normal"),
        ("Unexpected roaming charges added to my bill without international travel.", "High"),
        ("Auto-pay deducted money twice from my bank account.", "High"),
        ("Please explain the extra value-added service fee on my statement.", "Normal"),
    ],
    "SIM_Activation": [
        ("My new eSIM activation is stuck for 24 hours, number is dead.", "High"),
        ("Want to port my number from another operator to your network.", "Normal"),
        ("Lost my phone, please block my SIM immediately to prevent fraud.", "High"),
        ("KYC verification completed yesterday but SIM still shows no service.", "High"),
        ("How do I upgrade my physical SIM to an eSIM?", "Normal"),
    ],
    "Broadband_Tech": [
        ("Wi-Fi router red LOS light is blinking continuously, no internet.", "High"),
        ("Need technician visit to shift my fiber router to another room.", "Normal"),
        ("High ping and packet loss on fiber broadband connection.", "Normal"),
        ("Router power adapter stopped working, need replacement.", "High"),
        ("How do I change my broadband Wi-Fi password and SSID?", "Normal"),
    ],
}

# Extra noise words to make data realistic (so the model learns properly)
NOISE_PREFIXES = ["Hi team, ", "Urgent: ", "Hello, ", "Please help! ", ""]
NOISE_SUFFIXES = [" Ticket raised earlier.", " Kindly resolve ASAP.", " Thanks.", ""]

def generate_dataset(num_rows=600):
    rows = []
    departments = list(DATA_TEMPLATES.keys())

    for i in range(1, num_rows + 1):
        dept = random.choice(departments)
        text, urgency = random.choice(DATA_TEMPLATES[dept])

        # Add slight text variations
        full_text = f"{random.choice(NOISE_PREFIXES)}{text}{random.choice(NOISE_SUFFIXES)}"
        rows.append({
            "ticket_id": f"TKT-{1000 + i}",
            "complaint_text": full_text.strip(),
            "department": dept,
            "urgency": urgency
        })

    df = pd.DataFrame(rows)
    df.to_csv("telecom_complaints.csv", index=False)
    print(f"Saved {len(df)} tickets to telecom_complaints.csv!")
    print("\nSample rows:\n", df.head(5))

if __name__ == "__main__":
    generate_dataset()