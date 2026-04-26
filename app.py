# app.py - Main Flask Backend for AI Customer Feedback System
# Uses VADER Sentiment Analyzer to classify and score user feedback

from flask import Flask, render_template, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Flask application
app = Flask(__name__)

# Initialize VADER Sentiment Analyzer (once at startup for efficiency)
analyzer = SentimentIntensityAnalyzer()


def classify_sentiment(compound_score):
    """
    Classify sentiment label based on VADER compound score.
    VADER convention:
      compound >= 0.05  -> Positive
      compound <= -0.05 -> Negative
      otherwise         -> Neutral
    """
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route — handles both GET (show form) and POST (analyze feedback).
    On GET:  renders the empty feedback form.
    On POST: reads the submitted feedback, runs sentiment analysis,
             and returns the result back to the template.
    """
    result = None  # No result yet on initial page load

    if request.method == "POST":
        # Retrieve feedback text from the submitted form
        feedback_text = request.form.get("feedback", "").strip()

        if feedback_text:
            # Run VADER analysis — returns dict with pos, neg, neu, compound
            scores = analyzer.polarity_scores(feedback_text)

            compound = scores["compound"]
            label = classify_sentiment(compound)

            # Build result dict to pass to the template
            result = {
                "feedback": feedback_text,
                "label": label,
                "compound": round(compound, 4),
                "positive": round(scores["pos"], 4),
                "negative": round(scores["neg"], 4),
                "neutral": round(scores["neu"], 4),
            }

    # Render template — result is None on GET, populated on POST
    return render_template("index.html", result=result)


if __name__ == "__main__":
    # Run on localhost port 5000 in debug mode for development
    app.run(debug=True, host="127.0.0.1", port=5000)
