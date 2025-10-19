import pandas as pd
import random
from datetime import datetime, timedelta
import os

def generate_initial_dataset(path="data/chat_dataset.csv", n=100):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    users = ["Alice", "Bob", "Charlie", "David"]
    normal_msgs = [
        "Hey, how are you?", "Let's meet tomorrow.", "Happy birthday!", 
        "Call me later.", "See you at the cafe."
    ]
    suspicious_msgs = [
        "Send me your bank details.", "Click this link to claim your prize!",
        "Your account is compromised, login here!", "I can make you rich fast.",
        "Transfer money now or else!"
    ]

    data = []
    for _ in range(n):
        timestamp = datetime.now() - timedelta(days=random.randint(0,30), minutes=random.randint(0,1440))
        sender = random.choice(users)
        if random.random() < 0.3:
            msg = random.choice(suspicious_msgs)
            label = 1
        else:
            msg = random.choice(normal_msgs)
            label = 0
        data.append([timestamp, sender, msg, label])

    df = pd.DataFrame(data, columns=['timestamp','sender','message','label'])
    df.to_csv(path, index=False)
    print(f"✅ Dataset generated at {path}")
    return df


def main():
    generate_initial_dataset()

if __name__ == "__main__":
    main()