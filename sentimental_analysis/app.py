from flask import Flask, render_template, request, redirect, url_for
import requests

app = Flask(__name__)

# In-memory storage for comments
comments_storage = []

@app.route('/')
def enter_comment():
    return render_template('enter_comment.html')

@app.route('/submit_comment', methods=['POST'])
def submit_comment():
    comment = request.form['comment']
    comments_storage.append(comment)
    return redirect(url_for('view_comments'))

@app.route('/view_comments')
def view_comments():
    # Call the BentoML sentiment analysis API
    comments_with_sentiment = []
    for comment in comments_storage:
        response = requests.post("http://127.0.0.1:3000/predict_sentiment", json={"text": comment})
        sentiment = response.json().get("sentiment")
        comments_with_sentiment.append((comment, sentiment))

    return render_template('view_comments.html', comments=comments_with_sentiment)

if __name__ == '__main__':
    app.run(port=5003, debug=True)
