# AI Based Customer Feedback Analyzer

An AI-powered web application built using Flask and Natural Language Processing (NLP) techniques to analyze customer feedback and generate meaningful insights.

---

# Quick Start

```bash
# Install dependencies
pip install -r requirements.txt



# Run the application
python app.py

Open the application in your browser:

http://127.0.0.1:5000

Dashboard:

http://127.0.0.1:5000/dashboard


ai-feedback-system/
├── app.py                  # Flask backend application
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── database/
│   └── feedback.db         # SQLite database (auto-created)
├── templates/
│   ├── index.html          # Main feedback analysis page
│   └── dashboard.html      # Analytics dashboard
└── venv/                   # Virtual environment




| # | Feature                     | Description                                                                            |
| - | --------------------------- | -------------------------------------------------------------------------------------- |
| 1 | Sentiment Analysis          | Classifies feedback into Positive, Negative, or Neutral using VADER Sentiment Analyzer |
| 2 | Keyword Extraction          | Detects important words and common customer issues from feedback                       |
| 3 | Topic Classification        | Categorizes feedback into Product, Delivery, Service, and Pricing                      |
| 4 | Dashboard Visualization     | Displays analytics using interactive Chart.js graphs                                   |
| 5 | Feedback Summary Generation | Generates summary insights from customer reviews                                       |
| 6 | Trend Analysis              | Tracks sentiment trends over time                                                      |
| 7 | Filtering & Search          | Filters feedback by sentiment, category, and date                                      |
| 8 | Emotion Detection           | Detects emotions such as Happy, Angry, Frustrated, and Satisfied                       |



Technologies Used
Python
Flask
VADER Sentiment Analyzer
SQLite
HTML5
CSS3
JavaScript
Chart.js

Dependencies

Required packages:

flask>=2.3.0
vaderSentiment>=3.3.2

Install dependencies using:

pip install -r requirements.txt
Database

The project uses SQLite for storing customer feedback data.

The database file:

database/feedback.db

is automatically created when the application runs for the first time.



How the System Works:

User submits customer feedback through the web interface.
The Flask backend processes the feedback.
VADER Sentiment Analyzer calculates sentiment scores.
The system extracts keywords and classifies categories.
Emotion detection logic identifies user emotions.
Results are stored in the SQLite database.
Dashboard visualizations display analytics and trends.

=======================================================================

Future Improvements:

Integration with advanced AI/LLM models
Real-time analytics
User authentication system
Export reports as PDF/Excel
Cloud database deployment
Multi-language sentiment analysis


Author

Samrat Ghosh
