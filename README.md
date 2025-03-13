# Automatic Charity News Scraping Using Natural Language Processing

## Overview
This project automates the process of scraping, processing, and classifying charity-related news using web scraping and Natural Language Processing (NLP). The system collects charity-related articles from multiple sources, processes them using NLP techniques, and clusters them based on relevance before publishing verified content on a dedicated website.

## Features
- **Web Scraping**: Extracts news articles from websites and Twitter API.
- **Natural Language Processing (NLP)**: Cleans, processes, and extracts relevant content from news data.
- **Data Clustering**: Groups related articles together for better classification.
- **Manual Verification**: Ensures that only verified charity news is published.
- **Website Integration**: Publishes curated charity news articles online.

## Project Workflow
The system operates in three main modules:

### **Module 1: Web Scraping**
- Uses a scraping script to collect data from various news websites.
- Utilizes Twitter API to fetch relevant tweets.
- Converts the scraped data into structured CSV format for further processing.

### **Module 2: NLP Processing**
- Cleans and preprocesses extracted text data.
- Uses NLP techniques (via NLTK) to identify relevant keywords.
- Simplifies text for clustering in the next stage.

### **Module 3: Clustering & Verification**
- Groups similar articles using clustering techniques.
- Manually verifies data for accuracy.
- Publishes verified news to the website.

## Future Scope
- Improve and optimize the current system for better performance.
- Deploy the website on the web for public access.
- Develop a mobile app for better accessibility.
- Expand data sources by integrating additional news platforms.

## Requirements
To run this project, ensure you have the following installed:

```bash
pip install numpy pandas requests beautifulsoup4 nltk scikit-learn tweepy flask
```

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/charity-news-scraper.git
   cd charity-news-scraper
   ```
2. Configure the Twitter API credentials in the script.
3. Run the web scraping script to collect data:
   ```bash
   python scraper.py
   ```
4. Process the data using NLP:
   ```bash
   python nlp_processing.py
   ```
5. Perform clustering and verify the output:
   ```bash
   python clustering.py
   ```
6. Deploy the website:
   ```bash
   python app.py
   ```

## Contributors
- **Liya Derby**
- **Niranjan Krishnan**
- **Jees Jose**
- **Project Guide: Ravi Shankar**

## License
This project is open-source and licensed under the MIT License.

---
Feel free to contribute and enhance this project!

