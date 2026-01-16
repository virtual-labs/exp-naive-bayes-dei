/* Main Logic for Naive Bayes Experiment Simulation */

const stepsData = [
    {
        id: 'import_libraries',
        title: 'Importing Libraries',
        blocks: [
            {
                code: `<div class="output-success"># Importing necessary libraries</div>
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import HTML, display
print("Libraries Imported")`,
                output: `<div class="output-success">Libraries Imported</div>`
            }
        ]
    },
    {
        id: 'loading_dataset',
        title: 'Loading Dataset',
        blocks: [
            {
                code: `<div class="output-success"># Read the Spam SMS dataset</div>
data = pd.read_csv('spam_detection.csv', encoding='latin1', header=0, usecols=[0,1], engine="python")
print("Dataset loaded successfully")`,
                output: `<div class="output-text">Dataset loaded successfully </div>`
            }
        ]
    },
    {
        id: 'data_analysis',
        title: 'Data Analysis',
        blocks: [
            {
                code: `<div class="output-success"># Display first 5 rows</div>
data.head()`,
                output: `<table class="data-table">
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: left;">Category</th>
      <th style="text-align: left;">Message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">0</td>
      <td style="text-align: left;">spam</td>
      <td style="text-align: left;">
        Get your garden ready for summer with a free selection of summer bulbs and
        seeds worth £33.50, available only with The Scotsman this Saturday.
        To stop messages, visit notxt.co.uk.
      </td>
    </tr>
    <tr>
      <td style="text-align: left;">1</td>
      <td style="text-align: left;">spam</td>
      <td style="text-align: left;">
        This is your chance to be on a reality fantasy show.
        Call now at 08707509020. The cost is 20p per minute.
        NTT Ltd, PO Box 1327, Croydon CR9 5WB. 0870 is a national-rate call.
      </td>
    </tr>
    <tr>
      <td style="text-align: left;">2</td>
      <td style="text-align: left;">ham</td>
      <td style="text-align: left;">
        Love is not a decision; it is a feeling. If we could choose whom to love,
        life would be much simpler, but it would also be less magical.
      </td>
    </tr>
    <tr>
      <td style="text-align: left;">3</td>
      <td style="text-align: left;">ham</td>
      <td style="text-align: left;">
        Good friends care for each other, close friends understand each other,
        and true friends remain forever—beyond words and beyond time. Good night.
      </td>
    </tr>
    <tr>
      <td style="text-align: left;">4</td>
      <td style="text-align: left;">spam</td>
      <td style="text-align: left;">
        Get your garden ready for summer with a free selection of summer bulbs and
        seeds worth £33.50, available only with The Scotsman this Saturday.
        To stop messages, visit notxt.co.uk.
      </td>
    </tr>
  </tbody>
</table>
`
            },
            {
                code: `<div class="output-success"># Display Last 5 rows</div>
data.tail()`,
                output: `<table class="data-table">
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: left;">Category</th>
      <th style="text-align: left;">Message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">1737</td>
      <td style="text-align: left;">spam</td>
      <td style="text-align: left;">
        Do you want 750 anytime any network mins 150 text and a NEW VIDEO phone for only five pounds per week call 08002888812 or reply for delivery tomorrow
      </td>
    </tr>
    <tr>
      <td style="text-align: left;">1738</td>
      <td style="text-align: left;">ham</td>
      <td style="text-align: left;">
        Thanx...
      </td>
    </tr>
    <tr>
      <td style="text-align: left;">1739</td>
      <td style="text-align: left;">spam</td>
      <td style="text-align: left;">
        88800 and 89034 are premium phone services call 08718711108
      </td>
    </tr>
    <tr>
      <td style="text-align: left;">1740</td>
      <td style="text-align: left;">spam</td>
      <td style="text-align: left;">
        Today's Offer! Claim ur £150 worth of discount vouchers! Text YES to 85023 now! SavaMob, member offers mobile! T Cs 08717898035. £3.00 Sub. 16 . Unsub reply X
      </td>
    </tr>
    <tr>
      <td style="text-align: left;">1741</td>
      <td style="text-align: left;">ham</td>
      <td style="text-align: left;">
        am up to my eyes in philosophy
      </td>
    </tr>
  </tbody>
</table>
`
            },
            {
                code: `<div class="output-success"># Displays summary of dataset, including column names, data types, and non-null counts.</div>
data.info()`,
                output: `<pre class="output-text" style="background: transparent; border: none; font-family: inherit; margin: 0; padding: 0;">&lt;class 'pandas.core.frame.DataFrame'&gt;
RangeIndex: 1742 entries, 0 to 1741
Data columns (total 2 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Category  1742 non-null   object
 1   Message   1742 non-null   object
dtypes: object(2)
memory usage: 27.3+ KB</pre>`
            },
            {
                code: `<div class="output-success"># Checks the dataset for missing(NULL) values in each column</div>
data.isnull().sum()`,
                output: `<pre class="output-text" style="background: transparent; border: none; font-family: inherit; margin: 0; padding: 0;">Category    0
Message     0
dtype: int64</pre>`
            },
            {
                code: `<div class="output-success"># Shows the size of the dataset</div>
data.shape`,
                output: `<div class="output-text">(1742, 2)</div>`
            },
            {
                code: `<div class="output-success"># Counts the frequency of each unique category</div>
data['Category'].value_counts()`,
                output: `<table class="data-table" style="width: auto; min-width: 250px;">
  <thead>
    <tr>
      <th style="padding: 8px 15px;">Category</th>
      <th style="padding: 8px 15px;">Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px 15px;">Ham</td>
      <td style="padding: 8px 15px;">1003</td>
    </tr>
    <tr>
      <td style="padding: 8px 15px;">Spam</td>
      <td style="padding: 8px 15px;">739</td>
    </tr>
  </tbody>
</table>
<div class="output-text" style="margin-top: 10px;">dtype: int64</div>`
            }
        ]
    },
    {
        id: 'data_preprocessing',
        title: 'Data Preprocessing',
        blocks: [
            {
                code: `<div class="output-success"># Visualizes the distribution of spam and ham messages in the dataset</div>
category_counts = data['Category'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(category_counts, labels=['Ham (0)', 'Spam (1)'], 
    autopct='%1.1f%%', startangle=140, colors=['lightblue', 'salmon'])
plt.title('Distribution of Ham and Spam in Category'); 
plt.show()`,
                output: `<div class="output-header" style="text-align: left;">Distribution of Ham and Spam in Category</div>
<img src="images/spam-ham-distribution.png" alt="Distribution Pie Chart" style="max-width: 320px; width: 100%; height: auto; display: block; margin: 10px 0; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">`
            },
            {
                code: `<div class="output-success"># Sets up stemming and stop-word removal tools for text preprocessing.</div>
stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)
print("Initialized Porter Stemmer and English stopword list.")`,
                output: `<div class="output-text">Initialized Porter Stemmer and English stopword list.</div>`
            },
            {
                code: `<div class="output-success"># Performs basic text cleaning and normalisation</div>
def preprocess_text(s):
    if pd.isnull(s): return ""
    s = s.lower(); s = re.sub(r'http\S+|www\.\S+', ' ', s); s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip(); return s
print("Text Cleaned")`,
                output: `<div class="output-text">Text Cleaned</div>`
            }
        ]
    },
    {
        id: 'model_training',
        title: 'Model Training',
        blocks: [
            {
                code: `<div class="output-success"># Encodes spam labels and splits the data into stratified training and testing sets.</div>
data['label'] = (data['Category'].str.lower().str.strip() == 'spam').astype(int)
X_train, X_test, y_train, y_test = train_test_split( data['Message'], data['label'], 
    test_size=0.25, random_state=42, stratify=data['label'] )
print("Labels encoded and data split into training and testing sets.")`,
                output: `<div class="output-text">Labels encoded and data split into training and testing sets.</div>`
            },
            {
                code: `<div class="output-success"># Converts text messages into TF-IDF feature vectors.</div>
vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_features=50000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Text data vectorized using TF-IDF.")`,
                output: `<div class="output-text">Text data vectorized using TF-IDF.</div>`
            },
            {
                code: `<div class="output-success"># Trains a Multinomial Naive Bayes classifier on the TF-IDF features.</div>
model=MultinomialNB()
model.fit(X_train_vec,y_train)
print("Model training completed.")`,
                output: `<img src="images/naive .png" alt="MultinomialNB Estimator" style="width: 190px; height: auto; display: block; border: 1px solid #ced4da; border-radius: 4px;">
</div>
<div class="output-text">Model training completed.</div>
<div style="margin-top: 15px;">`
            }
        ]
    },
    {
        id: 'model_evaluation',
        title: 'Model Evaluation',
        blocks: [
            {
                code: `<div class="output-success"># Predicts on test data and evaluates model</div>
y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))`,
                output: `<pre class="output-text" style="background: transparent; border: none; font-family: inherit; margin: 0; padding: 0;">Accuracy: 0.9288990825688074

Classification Report:
              precision    recall  f1-score   support

      0(ham)       0.90      0.99      0.94       251
     1(spam)       0.99      0.84      0.91       185

    accuracy                           0.93       436
   macro avg       0.94      0.92      0.93       436
weighted avg       0.93      0.93      0.93       436
</pre>`
            },
            {
                code: `<div class="output-success"># Plots the confusion matrix with values to visualize correct and incorrect predictions</div>
cm = confusion_matrix(y_test, y_pred); plt.matshow(cm, cmap='Blues')
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, val, ha='center', va='center', color='white' if val > cm.max()/2 else 'black', fontsize=12)
plt.xticks([0,1], ['Pred Ham','Pred Spam'])
plt.yticks([0,1], ['Actual Ham','Actual Spam']); plt.colorbar();
plt.show(); print("Confusion matrix displayed.")`,
                output: `<div class="output-text">Confusion matrix displayed.</div>
<img src="images/confusion-matrix.png" alt="Confusion Matrix" style="max-width: 400px; display: block; margin: 10px 0; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">`
            },
            {
                code: `<div class="output-success"># Plots the Precision-Recall curve for the model</div>
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,4)); plt.plot(recall, precision); plt.title('Precision-Recall Curve'); plt.xlabel('Recall');plt.ylabel('Precision'); plt.tight_layout(); plt.show()
print("Precision-Recall curve displayed.")`,
                output: `<div class="output-text">Precision-Recall curve displayed.</div>
<img src="images/precision-recall-curve.png" alt="Precision-Recall Curve" style="max-width: 480px; display: block; margin: 10px 0; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">`
            },
            {
                code: `<div class="output-success"># Highlights words based on their contribution to spam/ham prediction using color intensity</div>
def color_text(words, contributions):
    max_val = max(abs(c) for c in contributions) or 1; html = ""
    for w, c in zip(words, contributions):
        intensity = int(255 * (abs(c) / max_val))
        if c > 0: color = f"rgb(255,{255-intensity},{255-intensity})"
        else: color = f"rgb({255-intensity},255,{255-intensity})"
        html += f"<span style='background-color:{color}; padding:3px; margin:2px; border-radius:4px;'>{w}</span> "
    return html
print("Text contribution visualization function defined.")`,
                output: `<div class="output-text">Text contribution visualization function defined.</div>`
            },
            {
                code: `<div class="output-success"># Computes and displays precision and recall of the model on test data</div>
def show_precision_recall(model, X_test_vec, y_test):
    preds = model.predict(X_test_vec)
    precision = precision_score(y_test, preds) ;recall = recall_score(y_test, preds)
print("Precision and recall calculation function ready.")`,
                output: `<div class="output-text">Precision and recall calculation function ready.</div>`
            },
            {
                code: `<div class="output-success"># Performs precision-recall evaluation and analyzes 'n' random test messages</div>
def analyze_multiple(model, vectorizer, X_test, y_test, n=5):
    show_precision_recall(model, X_test_vec, y_test)
    for i in range(n):
        analyze_random_message(model, vectorizer, X_test, y_test)
print("Multiple message analysis function defined.")`,
                output: `<div class="output-text">Multiple message analysis function defined.</div>`
            },
            {
                code: `<div class="output-success"># Runs evaluation and displays analysis of 5 random test messages</div>
analyze_multiple(model, vectorizer, X_test, y_test, n=5)
print(" MODEL PERFORMANCE ON TEST DATA")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
`,
                output: `<div class="output-text" style="font-family: monospace; line-height: 1.5;">
    MODEL PERFORMANCE ON TEST DATA<br>
    Precision: 0.9873<br>
    Recall   : 0.8432
</div>
<div style="margin-top: 15px; font-weight: bold; font-family: 'Outfit', sans-serif;">Choose one random test sample.</div>
<table class="data-table" style="width: 100%; border-collapse: collapse; margin-top: 10px; cursor: pointer;">
    <thead>
        <tr><th style="width: 80px; text-align: left;">Category</th><th style="text-align: left;">Message</th></tr>
    </thead>
    <tbody id="evalSamplesBody">
        <tr onclick="selectEvalSample(this, 0)"><td style="text-align: left;">ham</td><td style="text-align: left;">It's getting me down just waiting around.</td></tr>
        <tr onclick="selectEvalSample(this, 1)"><td style="text-align: left;">spam</td><td style="text-align: left;">Text and meet someone today. Reply with your name and age.</td></tr>
        <tr onclick="selectEvalSample(this, 2)"><td style="text-align: left;">ham</td><td style="text-align: left;">I.ll give her once i have it. Plus she said grinule greet you whenever we speak</td></tr>
        <tr onclick="selectEvalSample(this, 3)"><td style="text-align: left;">spam</td><td style="text-align: left;">Congratulations U can claim 2 VIP row A Tickets 2 C Blu in concert in November or Blu gift guaranteed Call 09061104276 to claim TS&Cs www.smsco.net cost£3.75max</td></tr>
        <tr onclick="selectEvalSample(this, 4)"><td style="text-align: left;">ham</td><td style="text-align: left;">thinking about you.</td></tr>
    </tbody>
</table>

<div id="evalSampleAnalysis" style="margin-top: 25px; display: none; padding: 20px; background: rgba(255,255,255,0.7); border-radius: 12px; border: 1px solid rgba(0,0,0,0.05); box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
    <div style="font-weight: bold; margin-bottom: 15px; font-size: 1.1em;">Message Analysis</div>
    <div id="analysisText" style="margin-bottom: 20px; line-height: 1.8;"></div>
    <div style="font-family: monospace; font-size: 0.95em;">
        <div><strong>Actual:</strong> <span id="actualLabel"></span></div>
        <div style="margin-top: 4px;"><strong>Predicted:</strong> <span id="predictedLabel" style="color: #ff4d4d; font-weight: bold;"></span></div>
        <div style="margin-top: 4px;"><strong>Spam Probability:</strong> <span id="spamProb"></span></div>
    </div>
</div>`
            }
        ]
    }
];

// Interactive evaluation logic
window.selectEvalSample = function (row, index) {
    // Highlight row
    const tbody = document.getElementById('evalSamplesBody');
    Array.from(tbody.getElementsByTagName('tr')).forEach(tr => {
        tr.style.backgroundColor = '';
        tr.style.color = '';
    });
    row.style.backgroundColor = '#d1e7dd'; // Light green

    const analysisContainer = document.getElementById('evalSampleAnalysis');
    const analysisText = document.getElementById('analysisText');
    const actualLabel = document.getElementById('actualLabel');
    const predictedLabel = document.getElementById('predictedLabel');
    const spamProb = document.getElementById('spamProb');

    analysisContainer.style.display = 'block';

    // Sample data for visualization
    const samples = [
        { cat: 'ham', msg: "It's getting me down just waiting around.", prob: 0.0408 },
        { cat: 'spam', msg: "Text and meet someone today. Reply with your name and age.", prob: 0.5772 },
        { cat: 'ham', msg: "I.ll give her once i have it. Plus she said grinule greet you whenever we speak", prob: 0.0322 },
        { cat: 'spam', msg: "Congratulations U can claim 2 VIP row A Tickets 2 C Blu in concert in November or Blu gift guaranteed Call 09061104276 to claim TS&Cs www.smsco.net cost£3.75max", prob: 0.9309 },
        { cat: 'ham', msg: "thinking about you.", prob: 0.0670 }
    ];

    const sample = samples[index];

    // Word influence mapping (simulated based on images)
    const wordInfluences = {
        // Spammy (Red)
        'text': 'spam_strong', 'reply': 'spam_strong', 'claim': 'spam_strong', 'call': 'spam_strong',
        'congratulations': 'spam_med', 'gift': 'spam_med', 'guaranteed': 'spam_med', 'cost': 'spam_med',
        'waiting': 'spam_weak', 'your': 'spam_weak', 'plus': 'spam_weak',

        // Hammy (Green)
        'getting': 'ham_strong', 'me': 'ham_strong', 'down': 'ham_strong', 'just': 'ham_strong',
        'meet': 'ham_strong', 'someone': 'ham_strong', 'name': 'ham_strong', 'age': 'ham_strong',
        'give': 'ham_strong', 'her': 'ham_strong', 'once': 'ham_strong', 'have': 'ham_strong',
        'she': 'ham_strong', 'said': 'ham_strong', 'greet': 'ham_strong', 'you': 'ham_strong',
        'whenever': 'ham_strong', 'we': 'ham_strong', 'speak': 'ham_strong',
        'thinking': 'ham_strong', 'about': 'ham_strong', 'and': 'ham_strong',
        'in': 'ham_med', 'can': 'ham_med'
    };

    // Simulate color_text highlighting
    const words = sample.msg.split(' ');
    analysisText.innerHTML = words.map(w => {
        const cleanW = w.toLowerCase().replace(/[^a-z0-9]/g, '');
        const influence = wordInfluences[cleanW];

        let color = 'rgba(0,0,0,0.05)'; // Default neutral

        if (influence === 'spam_strong') color = 'rgb(255, 0, 0)';      // Red
        else if (influence === 'spam_med') color = 'rgb(255, 150, 150)'; // Light Red/Pink
        else if (influence === 'spam_weak') color = 'rgb(255, 220, 220)'; // Very Light Red
        else if (influence === 'ham_strong') color = 'rgb(0, 255, 0)';   // Green
        else if (influence === 'ham_med') color = 'rgb(150, 255, 150)';  // Light Green

        // Override for image 0 specifically ("just" looks green, "waiting" looks red)
        if (cleanW === 'waiting') color = 'rgb(255, 220, 220)';

        return `<span style="background-color:${color}; padding:3px 6px; margin:2px; border-radius:4px; display: inline-block;">${w}</span>`;
    }).join(' ');

    actualLabel.innerText = sample.cat;
    predictedLabel.innerText = sample.cat;
    predictedLabel.style.color = sample.cat === 'spam' ? '#ff4d4d' : '#28a745';
    spamProb.innerText = sample.prob.toFixed(4);

    // Scroll to analysis
    analysisContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
};

// State Management
let STATE = {
    stepIndex: 0,
    subStepIndex: 0,
    stepsStatus: stepsData.map(() => ({ unlocked: false, completed: false, inProgress: false }))
};
STATE.stepsStatus[0].unlocked = true;

const stepsContainer = document.getElementById('stepsContainer');
const codeDisplay = document.getElementById('codeDisplay');
const outputDisplay = document.getElementById('outputDisplay');
const bottomPane = document.querySelector('.bottom-pane');
const runBtn = document.getElementById('runBtn');

function init() {
    renderSidebar();
    loadStep(0);
}

function renderSidebar() {
    stepsContainer.innerHTML = '';
    stepsData.forEach((step, index) => {
        const status = STATE.stepsStatus[index];
        const btn = document.createElement('button');
        btn.className = 'step-btn';
        if (status.unlocked) {
            if (status.completed) btn.classList.add('completed');
            else if (status.inProgress) btn.classList.add('in-progress');
            btn.onclick = () => loadStep(index);
            btn.innerHTML = status.completed ? `✓ ${step.title}` : `${index + 1}. ${step.title}`;
        } else {
            btn.classList.add('disabled');
            btn.innerHTML = `${index + 1}. ${step.title}`;
            btn.disabled = true;
        }
        if (index === STATE.stepIndex) btn.classList.add('active');
        stepsContainer.appendChild(btn);
    });

    const restartBtn = document.createElement('button');
    restartBtn.className = 'step-btn';
    restartBtn.style.cssText = "background:#333 !important; color:white; text-align:center;";
    restartBtn.innerText = "Restart Experiment";
    restartBtn.onclick = restartExperiment;
    stepsContainer.appendChild(restartBtn);

    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'step-btn';
    downloadBtn.style.cssText = "background:#F57C2A !important; color:white; margin-top:10px; text-align:center; display: flex; align-items: center; justify-content: center; gap: 10px;";
    downloadBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        Download Experiment`;
    downloadBtn.onclick = downloadPDF;
    stepsContainer.appendChild(downloadBtn);
}

function loadStep(index) {
    STATE.stepIndex = index;
    STATE.subStepIndex = 0;
    renderSidebar();
    updateUI();
}

const codeHeaderBar = document.querySelector('.code-header-bar');

function updateUI() {
    const step = stepsData[STATE.stepIndex];
    const block = step.blocks[STATE.subStepIndex];

    // Extract comment if present
    let finalCode = block.code;
    let commentText = "";

    // Check for the div wrapper first (as seen in data)
    const divRegex = /<div class="output-success">(.*?)<\/div>/;
    const match = finalCode.match(divRegex);

    if (match) {
        commentText = match[1]; // Get the text inside tags
        finalCode = finalCode.replace(match[0], '').trim(); // Remove the comment line and trim
        codeHeaderBar.style.display = 'block';
        codeHeaderBar.innerText = commentText;
    } else {
        codeHeaderBar.style.display = 'none';
        codeHeaderBar.innerText = '';
    }

    codeDisplay.innerHTML = highlightCode(finalCode);
    bottomPane.classList.remove('active-output');
    outputDisplay.innerHTML = '<div class="placeholder-text">Click the Run button to execute...</div>';
    runBtn.style.display = 'flex';
    runBtn.classList.remove('completed');
    runBtn.style.backgroundColor = '#F57C2A'; // Orange for Run
    runBtn.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
    runBtn.onclick = runStep;
}

function runStep() {
    const step = stepsData[STATE.stepIndex];
    const block = step.blocks[STATE.subStepIndex];

    // 1. Loading State
    outputDisplay.innerHTML = '<div class="loading-spinner">Running code...</div>';
    runBtn.disabled = true;

    // 2. Simulated Delay
    setTimeout(() => {
        // 3. Show Output
        outputDisplay.innerHTML = block.output;
        bottomPane.classList.add('active-output');

        // 4. Update Button State to Checkmark (Success)
        runBtn.classList.add('completed');
        runBtn.style.backgroundColor = '#A6CE63'; // Green (#A6CE63)
        runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';

        // Mark partial progress
        STATE.stepsStatus[STATE.stepIndex].inProgress = true;
        renderSidebar();

        // Check if this is the Random Prediction block
        if (document.getElementById('randomPredTableBody')) {
            window.generateRandomPrediction && window.generateRandomPrediction();
        }

        // 5. Handle Next Logic
        const hasNextBlock = STATE.subStepIndex < step.blocks.length - 1;

        if (hasNextBlock) {
            // Wait 1s then change button to "Next"
            setTimeout(() => {
                runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                runBtn.style.backgroundColor = '#5FA8E4'; // Orange
                runBtn.disabled = false;

                // Switch handler to Next
                runBtn.onclick = () => { STATE.subStepIndex++; updateUI(); };
            }, 500);

        } else {
            // Step Fully Completed
            STATE.stepsStatus[STATE.stepIndex].completed = true;
            renderSidebar(); // Update Current Step to Green Immediately

            // Unlock next step logic
            if (STATE.stepIndex < stepsData.length - 1) {
                STATE.stepsStatus[STATE.stepIndex + 1].unlocked = true;
                renderSidebar(); // Update Next Step to Red Immediately

                // Manual Next Step Arrow Button
                setTimeout(() => {
                    // Change button to Blue Arrow for Next Step
                    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                    runBtn.style.backgroundColor = '#5FA8E4'; // Blue (#5FA8E4)
                    runBtn.disabled = false;

                    // Logic to go to next MAIN step
                    runBtn.onclick = function () {
                        loadStep(STATE.stepIndex + 1);
                    };
                }, 500);
            } else {
                // End of Experiment - Show "Next" (Finish) Button
                renderSidebar();
                setTimeout(() => {
                    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                    runBtn.style.backgroundColor = '#72b2f7ff'; // Orange
                    runBtn.disabled = false;
                    runBtn.onclick = showCompletionMessage;
                }, 500);
            }
        }

    }, 500);
}

function highlightCode(code) {
    if (!code) return "";
    return code
        .replace(/import /g, '<span class="kw">import </span>')
        .replace(/from /g, '<span class="kw">from </span>')
        .replace(/print/g, '<span class="func">print</span>')
        .replace(/pd\./g, '<span class="func">pd.</span>')
        .replace(/\.fit/g, '<span class="func">.fit</span>')
        .replace(/\.predict/g, '<span class="func">.predict</span>');
}

window.generateRandomPrediction = function () {
    const tbody = document.getElementById('randomPredTableBody');
    if (!tbody) return;
    const samples = [
        { id: 10, msg: "Working on the project now.", label: 0, pred: 0 },
        { id: 25, msg: "WINNER! You won a FREE gift!", label: 1, pred: 1 },
        { id: 54, msg: "Lunch at 1pm?", label: 0, pred: 0 }
    ];
    tbody.innerHTML = samples.map(s => `
    <tr onclick='window.showPredictionResult(${JSON.stringify(s)})'>
      <td>${s.id}</td><td>${s.msg}</td><td>${s.label ? 'Spam' : 'Ham'}</td>
    </tr>
  `).join('');
};

window.showPredictionResult = function (s) {
    const res = document.getElementById('randomPredResult');
    res.style.display = 'block';
    res.innerHTML = `<strong>Result:</strong><br>Message: "${s.msg}"<br>Actual: ${s.label ? 'Spam' : 'Ham'}<br>Predicted: ${s.pred ? 'Spam' : 'Ham'}<br>Status: <span style="color:green">Correct ✅</span>`;
};

function showCompletionMessage() {
    outputDisplay.innerHTML = `<div style="text-align: center; animation: fadeIn 1s ease;">
      <h1 style="color: #2a9d8f; font-size: 2.5rem; margin-bottom: 20px;">Experiment Completed! ✔️</h1>
      <p style="font-size: 1.5rem; color: #333;">You have completed naive bayes classification successfully!</p>
      <button onclick="location.reload()" style="margin-top: 30px; padding: 15px 30px; background-color: #f7a072; color: white; border: none; border-radius: 10px; font-size: 1.2rem; cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">Restart Experiment</button>
    </div>`;
    runBtn.style.display = 'none';
}

function restartExperiment() {
    location.reload();
}

function downloadPDF() {
    const link = document.createElement('a');
    link.href = './Exp-Naive-Bayes.pdf';
    link.download = 'Exp-Naive-Bayes.pdf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

init();
