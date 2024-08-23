from flask import Flask, render_template, request , jsonify

import google.generativeai as genai

from sklearn.linear_model import LinearRegression
import pandas as pd

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter



app = Flask(__name__)
app.secret_key = 'your_secret_key'


API_KEY = 'AIzaSyCnHiPnc81WluNjSklL6lLR5FO_NbHRCfM'
#'AIzaSyCCrYnLhDIgToWeG4u_nPpQcB9uNJMze0U'
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

medical_keywords = [
    'doctor', 'medicine', 'health', 'symptom', 'treatment', 'diagnosis', 
    'therapy', 'medical', 'hospital', 'clinic', 'pharmacy', 'nurse', 
    'emergency', 'surgery', 'physician', 'prescription', 'patient', 
    'healthcare', 'pediatrician', 'dermatologist', 'gynecologist', 
    'cardiologist', 'neurologist', 'oncologist', 'radiologist', 
    'psychiatrist', 'ophthalmologist', 'orthopedic', 'dietitian', 
    'allergist', 'chiropractor', 'podiatrist', 'medication', 'delivery', 
    'order', 'track', 'shipment', 'customer service', 'pharmacy network', 
    'health advice', 'emergency assistance', 'drug recall', 'side effects', 
    'health tips', 'medication reminder', 'privacy', 'compliance', 'regulation', 
    'data privacy', 'healthcare provider', 'first aid', 'health guide', 
    'medicine availability', 'online pharmacy', 'prescription refill', 
    'pharmacy support', 'medication information', 'drug interaction', 
    'drug safety', 'medical emergency', 'pharmacy services', 'drug delivery', 
    'medical delivery', 'patient support', 'order status', 'payment options', 
    'drug compatibility', 'pharmaceutical care', 'patient care', 'medicine use', 
    'healthcare advice', 'prescription advice', 'medication order', 'prescription order', 
    'medication guidance', 'pharmacy assistance', 'healthcare support',
    'consultation', 'doctor consultation', 'medical advice', 'health consultation', 
    'telemedicine', 'virtual consultation', 'medical specialist', 'doctor appointment', 
    'online doctor', 'specialist consultation', 'second opinion', 'health specialist', 
    'medical consultation', 'physician consultation', 'GP consultation', 'doctor visit', 
    'health check', 'medical opinion', 'medical referral', 'remote consultation',
    'lab test', 'blood test', 'urine test', 'diagnostic test', 'pathology', 
    'laboratory', 'lab technician', 'test result', 'lab procedure', 
    'medical test', 'clinical test', 'biopsy', 'culture test', 'genetic test', 
    'microbiology test', 'serology test', 'immunology test', 'radiology test', 
    'PCR test', 'MRI scan', 'CT scan', 'X-ray', 'ultrasound' , 'Hi'
]


csv_path = 'datasetFile.csv'  
data = pd.read_csv(csv_path)

X = data[['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5', 'Parameter 6', 'Parameter 7', 'Parameter 8']]
y = data['Parameter 9']
X = X.to_numpy()

model1 = LinearRegression()
model1.fit(X, y)

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/CV')
def CV():
    return render_template('CV.html')


@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        inputs = [float(request.form[field]) for field in ['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5', 'Parameter 6', 'Parameter 7', 'Parameter 8']]
        prediction = model1.predict([inputs])
        output = "Argument 1" if prediction[0] >= 0.5 else "Argument 2"
        return render_template('index1.html', prediction_text=f'{output}')
    return render_template('index1.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chat.html')

@app.route('/coding')
def coding():
    return render_template('coding.html')





@app.route('/ask', methods=['POST'])
def ask():
    user_message = str(request.form['messageText'])
    
    if not is_medical_query(user_message):
        bot_response_text = "I'm sorry, I can only answer medical-related questions. Please ask a question related to medical topics."
    else:
        bot_response = chat.send_message(user_message)
        bot_response_text = bot_response.text
    
    return jsonify({'status': 'OK', 'answer': bot_response_text})

def is_medical_query(query):
    return any(keyword.lower() in query.lower() for keyword in medical_keywords)

# Route to display the .py file
@app.route('/chatbotScript')
def chatbotScript():
    with open('static/chatbot.py', 'r') as f:
        code = f.read()
    lexer = PythonLexer()
    formatter = HtmlFormatter(full=True, linenos=True, style='friendly')
    highlighted_code = highlight(code, lexer, formatter)
    html_content = f"""
    <html>
    <head>
        <title>Chatbot Script</title>
        <style>{formatter.get_style_defs('.highlight')}</style>
    </head>
    <body>
        <h1>Highlighted Python Script</h1>
        <div class="highlight">{highlighted_code}</div>
    </body>
    </html>
    """
    return html_content





if __name__ == '__main__':
    app.run(debug=True,port=9007)
