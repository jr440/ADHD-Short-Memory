from flask import Flask, render_template, request, jsonify, Response
import sqlite3
import time

app = Flask(__name__)

def query_database(query, params=()):
    """Helper function to query the SQLite database."""
    conn = sqlite3.connect('memory.db')
    cursor = conn.cursor()
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    return results

@app.route('/')
def home():
    """Serves the main webpage."""
    return render_template('frontend.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Answers user questions by querying the database."""
    user_question = request.json.get('question')
    
    # Example logic to query transcripts for relevant information
    transcripts = query_database("SELECT content FROM transcripts")
    all_text = " ".join([row[0] for row in transcripts])

    # Here, you can use AI logic to process the question and find an answer
    ai_answer = f"You asked: '{user_question}'. The database contains: '{all_text[:100]}...'"
    
    return jsonify({'answer': ai_answer})

@app.route('/transcripts', methods=['GET'])
def get_transcripts():
    """Fetches all transcripts from the database."""
    transcripts = query_database("SELECT timestamp, content FROM transcripts")
    return jsonify({'transcripts': [{'timestamp': row[0], 'content': row[1]} for row in transcripts]})

@app.route('/summary', methods=['GET'])
def get_summary():
    """Fetches the latest hourly summary."""
    summaries = query_database("SELECT timestamp, content FROM transcripts ORDER BY timestamp DESC LIMIT 1")
    if summaries:
        return jsonify({'summary': {'timestamp': summaries[0][0], 'content': summaries[0][1]}})
    return jsonify({'summary': None})

@app.route('/sse', methods=['GET'])
def sse():
    """Server-Sent Events endpoint for real-time updates."""
    def generate():
        last_summary = None
        while True:
            summaries = query_database("SELECT timestamp, content FROM transcripts ORDER BY timestamp DESC LIMIT 1")
            if summaries and summaries[0] != last_summary:
                last_summary = summaries[0]
                yield f"data: {summaries[0][1]}\n\n"
            time.sleep(5)
    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)