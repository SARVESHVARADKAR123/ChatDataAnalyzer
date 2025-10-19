import pandas as pd
import re
from typing import List, Dict

class ChatParser:
    @staticmethod
    def parse_whatsapp(file_path: str) -> pd.DataFrame:
        """
        Parse WhatsApp chat export text file
        
        Args:
            file_path (str): Path to WhatsApp chat export
        
        Returns:
            pd.DataFrame: Parsed chat messages
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        messages = []
        for line in lines:
            # WhatsApp export format: [DD/MM/YYYY, HH:MM:SS] Sender: Message
            match = re.match(r'\[(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}:\d{2})\] (.*?): (.*)', line)
            if match:
                date, time, sender, message = match.groups()
                messages.append({
                    'timestamp': f'{date} {time}',
                    'sender': sender,
                    'message': message.strip()
                })
        
        return pd.DataFrame(messages)
    
    @staticmethod
    def parse_telegram(file_path: str) -> pd.DataFrame:
        """
        Parse Telegram chat export JSON file
        
        Args:
            file_path (str): Path to Telegram chat export
        
        Returns:
            pd.DataFrame: Parsed chat messages
        """
        # Placeholder for Telegram parsing logic
        # Actual implementation would depend on Telegram's export format
        df = pd.read_json(file_path)
        return df[['date', 'from', 'text']].rename(columns={
            'date': 'timestamp', 
            'from': 'sender', 
            'text': 'message'
        })
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Basic text preprocessing
        
        Args:
            text (str): Input message text
        
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text.strip()

def main():
    # Example usage
    whatsapp_df = ChatParser.parse_whatsapp('data/whatsapp_chat.txt')
    whatsapp_df['processed_message'] = whatsapp_df['message'].apply(ChatParser.preprocess_text)
    whatsapp_df.to_csv('data/processed_whatsapp_chat.csv', index=False)

if __name__ == '__main__':
    main()
