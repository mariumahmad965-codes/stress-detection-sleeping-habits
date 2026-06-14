import streamlit as st
import pickle, io, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from fpdf import FPDF
from pathlib import Path

# ── Load model artifacts ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
model    = pickle.load(open(BASE_DIR / 'stress_model.pkl',    'rb'))
scaler   = pickle.load(open(BASE_DIR / 'stress_scaler.pkl',   'rb'))
features = pickle.load(open(BASE_DIR / 'stress_features.pkl', 'rb'))

# ── Load dataset for population comparison ────────────────────────────────────
@st.cache_data
def load_dataset():
    df = pd.read_csv(BASE_DIR / 'SaYoPillow.csv')
    df.rename(columns={
        'sr':'snoring_rate','rr':'respiration_rate','t':'body_temperature',
        'lm':'limb_movement','bo':'blood_oxygen','rem':'rem_sleep',
        'sr.1':'sleep_hours','hr':'heart_rate','sl':'stress_level'
    }, inplace=True)
    return df

df_data = load_dataset()

# ── Constants ─────────────────────────────────────────────────────────────────
STRESS_INFO = {
    0: {'label':'Low',         'emoji':'😊','color':'#e8f5e9','border':'#2e7d32','text':'#1b5e20',
        'description':'Your sleep physiological signals look healthy. You are in a calm, rested state.',
        'bar_color':'#4CAF50'},
    1: {'label':'Low-Medium',  'emoji':'🙂','color':'#fffde7','border':'#f9a825','text':'#e65100',
        'description':'Mild stress indicators detected. Your body shows minor sleep disturbances.',
        'bar_color':'#FFEB3B'},
    2: {'label':'Medium',      'emoji':'😐','color':'#fff3e0','border':'#ef6c00','text':'#bf360c',
        'description':'Moderate stress detected. Several physiological signals are outside the optimal range.',
        'bar_color':'#FF9800'},
    3: {'label':'Medium-High', 'emoji':'😟','color':'#fce4ec','border':'#c62828','text':'#b71c1c',
        'description':'High stress detected. Multiple signals indicate your body is under significant strain.',
        'bar_color':'#F44336'},
    4: {'label':'High',        'emoji':'😰','color':'#ffebee','border':'#b71c1c','text':'#7f0000',
        'description':'Critical stress level. Your sleep data shows severe physiological stress markers.',
        'bar_color':'#B71C1C'},
}

NORMAL_RANGES = {
    'snoring_rate':     (0,   40,  'Low snoring = better airway'),
    'respiration_rate': (12,  18,  '12-18 breaths/min is ideal'),
    'body_temperature': (96.8,99,  '96.8-99 degreesF during sleep'),
    'limb_movement':    (0,   8,   'Less movement = deeper sleep'),
    'blood_oxygen':     (95,  100, '>=95% is healthy'),
    'rem_sleep':        (20,  25,  '20-25% of sleep should be REM'),
    'sleep_hours':      (7,   9,   '7-9 hours recommended'),
    'heart_rate':       (40,  60,  '40-60 BPM at rest is ideal'),
}

FEATURE_DESCRIPTIONS = {
    'snoring_rate':     'Snoring Rate (%)',
    'respiration_rate': 'Respiration Rate (breaths/min)',
    'body_temperature': 'Body Temperature ( degreesF)',
    'limb_movement':    'Limb Movement Rate',
    'blood_oxygen':     'Blood Oxygen Level (%)',
    'rem_sleep':        'REM Sleep (%)',
    'sleep_hours':      'Sleep Hours',
    'heart_rate':       'Heart Rate (BPM)',
}

# ── Helper Functions ──────────────────────────────────────────────────────────
def predict(vals):
    df_in = pd.DataFrame([vals])
    df_in['oxygen_heart_ratio']  = df_in['blood_oxygen'] / df_in['heart_rate']
    df_in['sleep_quality_score'] = df_in['rem_sleep'] * df_in['sleep_hours']
    df_in['stress_indicator']    = df_in['snoring_rate'] + df_in['limb_movement'] + df_in['respiration_rate']
    df_in['autonomic_index']     = df_in['heart_rate'] / df_in['blood_oxygen']
    df_in['recovery_score']      = (df_in['blood_oxygen'] * df_in['rem_sleep']) / df_in['heart_rate']
    df_in['sleep_disruption']    = df_in['snoring_rate'] * df_in['limb_movement'] / 100
    scaled = scaler.transform(df_in[features])
    return int(model.predict(scaled)[0])

def get_stress_proba(vals):
    """Get probability for each stress class."""
    df_in = pd.DataFrame([vals])
    df_in['oxygen_heart_ratio']  = df_in['blood_oxygen'] / df_in['heart_rate']
    df_in['sleep_quality_score'] = df_in['rem_sleep'] * df_in['sleep_hours']
    df_in['stress_indicator']    = df_in['snoring_rate'] + df_in['limb_movement'] + df_in['respiration_rate']
    df_in['autonomic_index']     = df_in['heart_rate'] / df_in['blood_oxygen']
    df_in['recovery_score']      = (df_in['blood_oxygen'] * df_in['rem_sleep']) / df_in['heart_rate']
    df_in['sleep_disruption']    = df_in['snoring_rate'] * df_in['limb_movement'] / 100
    scaled = scaler.transform(df_in[features])
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(scaled)[0]
    # fallback: one-hot
    pred = int(model.predict(scaled)[0])
    p = [0.0]*5; p[pred] = 1.0
    return p

def get_abnormal_readings(v):
    issues = []
    if v['snoring_rate']     > 40  : issues.append(('Snoring Rate',     'HIGH', 'May indicate sleep apnea or airway obstruction'))
    if v['respiration_rate'] > 18  : issues.append(('Respiration Rate', 'HIGH', 'Elevated breathing rate - body under stress'))
    if v['respiration_rate'] < 12  : issues.append(('Respiration Rate', 'LOW',  'Very slow breathing rate'))
    if v['body_temperature'] < 96.8: issues.append(('Body Temperature', 'LOW',  'Below normal range during sleep'))
    if v['body_temperature'] > 99  : issues.append(('Body Temperature', 'HIGH', 'Fever range - immune system active'))
    if v['limb_movement']    > 8   : issues.append(('Limb Movement',    'HIGH', 'Restless legs - disrupting deep sleep'))
    if v['blood_oxygen']     < 95  : issues.append(('Blood Oxygen',     'LOW',  'Low oxygen saturation - risk of hypoxia'))
    if v['rem_sleep']        < 20  : issues.append(('REM Sleep',        'LOW',  'Insufficient REM - affects memory & mood'))
    if v['sleep_hours']      < 7   : issues.append(('Sleep Hours',      'LOW',  'Sleep deprivation detected'))
    if v['sleep_hours']      > 9   : issues.append(('Sleep Hours',      'HIGH', 'Excessive sleep - may indicate fatigue disorder'))
    if v['heart_rate']       > 60  : issues.append(('Heart Rate',       'HIGH', 'Elevated resting heart rate during sleep'))
    return issues

def calculate_sleep_health_score(v):
    score = 100
    sh = v['sleep_hours']
    if   7<=sh<=9  : pass
    elif 6<=sh<7   : score -= 10
    elif 5<=sh<6   : score -= 22
    else           : score -= 38
    bo = v['blood_oxygen']
    if   bo>=95: pass
    elif bo>=90: score -= 12
    elif bo>=85: score -= 22
    else       : score -= 32
    hr = v['heart_rate']
    if   40<=hr<=60: pass
    elif hr<=70    : score -= 8
    elif hr<=80    : score -= 18
    else           : score -= 28
    sr = v['snoring_rate']
    if   sr<20: pass
    elif sr<40: score -= 5
    elif sr<70: score -= 12
    else      : score -= 20
    rem = v['rem_sleep']
    if   20<=rem<=25: pass
    elif rem>=15    : score -= 5
    elif rem>=10    : score -= 12
    else            : score -= 18
    lm = v['limb_movement']
    if   lm<5 : pass
    elif lm<10: score -= 5
    else      : score -= 12
    return max(0, min(100, score))

def get_doctor_recommendations(stress_level, issues):
    recs = []
    if stress_level >= 3:
        recs.append(("🏥 Sleep Medicine Specialist", "Multiple high-stress markers detected. A formal sleep study (polysomnography) is strongly recommended."))
        recs.append(("🩺 General Physician",          "High physiological stress. A full blood panel and health check-up is advised."))
    if any('Blood Oxygen' in i[0] for i in issues):
        recs.append(("🫁 Pulmonologist / Sleep Apnea Specialist", "Low SpO2 during sleep is a key sign of sleep apnea. A CPAP evaluation may help."))
    if any('Heart Rate'   in i[0] for i in issues):
        recs.append(("❤️ Cardiologist",  "Elevated heart rate during sleep. An ECG and cardiac evaluation is recommended."))
    if any('Limb Movement'in i[0] for i in issues):
        recs.append(("🧠 Neurologist",   "High limb movement may indicate Restless Leg Syndrome (RLS) or Periodic Limb Movement Disorder."))
    if any('Body Temp'    in i[0] for i in issues):
        recs.append(("🌡️ General Physician","Abnormal body temperature during sleep. Rule out infection or thyroid disorder."))
    if stress_level <= 1 and not issues:
        recs.append(("✅ No Specialist Needed", "Your readings are within healthy range. Maintain your current sleep routine!"))
    if not recs:
        recs.append(("💊 General Physician", "A few minor deviations detected. A routine check-up is advisable."))
    return recs

# ── Chart: Probability Bar ────────────────────────────────────────────────────
def plot_stress_probability(probas):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    labels = ['Low (0)', 'Low-Med (1)', 'Medium (2)', 'Med-High (3)', 'High (4)']
    colors = ['#4CAF50','#FFEB3B','#FF9800','#F44336','#B71C1C']
    bars = ax.barh(labels, [p*100 for p in probas], color=colors, height=0.6, edgecolor='white')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)', fontsize=10)
    ax.set_title('Stress Level Probability Distribution', fontsize=12, fontweight='bold')
    for bar, prob in zip(bars, probas):
        if prob > 0.03:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.tight_layout()
    return fig

# ── Chart: Radar / Feature Status ────────────────────────────────────────────
def plot_feature_gauge(vals, issues):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    issue_keys = [i[0] for i in issues]
    
    DISPLAY = {
        'snoring_rate':     ('Snoring Rate', '%',    0, 100,  0,  40),
        'respiration_rate': ('Respiration',  'b/m', 10,  30, 12,  18),
        'body_temperature': ('Body Temp',    ' degreesF',  85, 100, 96.8,99),
        'limb_movement':    ('Limb Move',    '',     0,  20,  0,   8),
        'blood_oxygen':     ('Blood O2',    '%',   80, 100, 95, 100),
        'rem_sleep':        ('REM Sleep',   '%',    0, 105, 20,  25),
        'sleep_hours':      ('Sleep Hrs',   'hrs',  0,   9,  7,   9),
        'heart_rate':       ('Heart Rate',  'BPM', 40,  85, 40,  60),
    }
    
    for i, (key, (title, unit, vmin, vmax, norm_lo, norm_hi)) in enumerate(DISPLAY.items()):
        ax = axes[i]
        val = vals[key]
        
        # Background zones
        full_range = vmax - vmin
        norm_start = (norm_lo - vmin) / full_range
        norm_width = (norm_hi - norm_lo) / full_range
        
        ax.barh(0, 1, color='#ffcdd2', height=0.5)   # danger zone
        ax.barh(0, norm_width, left=norm_start, color='#c8e6c9', height=0.5)  # normal zone
        
        # User value marker
        val_pos = max(0, min(1, (val - vmin) / full_range))
        is_abnormal = title.replace('2','2') in [i[0] for i in issues] or key.replace('_',' ').title() in [i[0] for i in issues]
        marker_color = '#F44336' if is_abnormal else '#1976D2'
        ax.plot(val_pos, 0, 'v', markersize=14, color=marker_color, zorder=5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.6)
        ax.set_title(f'{title}\n{val:.1f} {unit}', fontsize=9, fontweight='bold',
                     color=marker_color)
        ax.axis('off')
        
        # Min / max labels
        ax.text(0,   -0.38, str(vmin), ha='left',   fontsize=7, color='gray')
        ax.text(1,   -0.38, str(vmax), ha='right',  fontsize=7, color='gray')
        ax.text(norm_start+norm_width/2, -0.38, f'Normal\n{norm_lo}-{norm_hi}',
                ha='center', fontsize=6.5, color='#388E3C')
    
    fig.suptitle('Your Readings vs Normal Range  (🟢 Normal Zone | 🔴 Risk Zone | ▼ Your Value)',
                 fontsize=10, y=0.98)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

# ── Chart: Population Comparison ─────────────────────────────────────────────
def plot_population_comparison(vals, stress_level):
    same_stress = df_data[df_data['stress_level'] == stress_level]
    features_to_show = ['snoring_rate','sleep_hours','blood_oxygen','heart_rate']
    labels = ['Snoring Rate', 'Sleep Hours', 'Blood Oxygen', 'Heart Rate']
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    colors_map = ['#FF7043','#66BB6A','#42A5F5','#AB47BC']
    
    for i, (feat, label, color) in enumerate(zip(features_to_show, labels, colors_map)):
        ax = axes[i]
        ax.hist(same_stress[feat], bins=15, color=color, alpha=0.6, edgecolor='white', label='Same Stress Group')
        ax.axvline(vals[feat], color='black', linewidth=2.5, linestyle='--', label=f'You: {vals[feat]:.1f}')
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Value', fontsize=8)
        ax.legend(fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle(f'Your Values vs Others with Same Stress Level ({STRESS_INFO[stress_level]["label"]})',
                 fontsize=11, fontweight='bold')
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

# ── Chart: Feature Importance ─────────────────────────────────────────────────
def plot_feature_importance():
    # Use the base estimators of the voting classifier
    importance_map = {}
    for est in model.estimators_:
        # estimators_ can be a list of estimators OR list of (name, est) tuples
        if isinstance(est, tuple):
            est = est[1]
        if hasattr(est, 'feature_importances_'):
            for f, imp in zip(features, est.feature_importances_):
                importance_map[f] = importance_map.get(f, 0) + imp

    if not importance_map:
        return None
    
    # Show top 8 features
    imp_series = pd.Series(importance_map).sort_values(ascending=True)
    imp_series = imp_series.tail(8)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#EF5350' if v > imp_series.mean() else '#42A5F5' for v in imp_series.values]
    bars = ax.barh(imp_series.index, imp_series.values, color=colors, edgecolor='white')
    ax.set_title('Feature Importance in Stress Prediction\n(Red = High Impact Features)', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

# ── Chart: Stress Trend Simulation ───────────────────────────────────────────
def plot_stress_trend(vals):
    """Show how stress changes if sleep hours improve."""
    sleep_range = np.linspace(0, 9, 20)
    stress_preds = []
    for sh in sleep_range:
        v = vals.copy()
        v['sleep_hours'] = sh
        stress_preds.append(predict(v))
    
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ['#4CAF50','#FFEB3B','#FF9800','#F44336','#B71C1C']
    for j in range(5):
        ax.axhspan(j - 0.5, j + 0.5, alpha=0.08, color=colors[j])
    
    ax.plot(sleep_range, stress_preds, 'o-', color='#1565C0', linewidth=2.5, markersize=5)
    ax.axvline(vals['sleep_hours'], color='red', linestyle='--', linewidth=2, label=f'Your sleep: {vals["sleep_hours"]:.1f} hrs')
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(['Low','Low-Med','Medium','Med-High','High'], fontsize=9)
    ax.set_xlabel('Sleep Hours', fontsize=10)
    ax.set_title('How Would More Sleep Change Your Stress Level?', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.6, 4.6)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

def plot_oxygen_trend(vals):
    """Show how blood oxygen level affects stress."""
    oxy_range = np.linspace(82, 97, 20)
    stress_preds = []
    for bo in oxy_range:
        v = vals.copy()
        v['blood_oxygen'] = bo
        stress_preds.append(predict(v))
    
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ['#4CAF50','#FFEB3B','#FF9800','#F44336','#B71C1C']
    for j in range(5):
        ax.axhspan(j - 0.5, j + 0.5, alpha=0.08, color=colors[j])
    
    ax.plot(oxy_range, stress_preds, 's-', color='#7B1FA2', linewidth=2.5, markersize=5)
    ax.axvline(vals['blood_oxygen'], color='#E65100', linestyle='--', linewidth=2,
               label=f'Your SpO2: {vals["blood_oxygen"]:.1f}%')
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(['Low','Low-Med','Medium','Med-High','High'], fontsize=9)
    ax.set_xlabel('Blood Oxygen (%)', fontsize=10)
    ax.set_title('How Blood Oxygen Level Affects Your Stress Prediction', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(82, 97)
    ax.set_ylim(-0.6, 4.6)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

# ── PDF Report ────────────────────────────────────────────────────────────────

def clean_text(text):
    """Remove all non-ASCII characters for PDF compatibility."""
    return ''.join(c if ord(c) < 128 else '' for c in str(text)).strip()

def generate_pdf(name, age, gender, vals, stress_level, health_score, issues, recs, probas):
    pdf = FPDF()
    pdf.add_page()
    # Header
    pdf.set_fill_color(26,35,126)
    pdf.rect(0,0,210,38,'F')
    pdf.set_text_color(255,255,255)
    pdf.set_font('Helvetica','B',20)
    pdf.set_xy(0,7)
    pdf.cell(210,10,'Sleep Stress Analysis Report',align='C',ln=True)
    pdf.set_font('Helvetica','',11)
    pdf.cell(210,8,'AI-Powered Stress Detection | SaYoPillow IEEE Dataset',align='C',ln=True)
    pdf.cell(210,7,f'Generated: {datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")}',align='C',ln=True)
    pdf.set_text_color(0,0,0)
    pdf.ln(5)

    # Patient
    pdf.set_font('Helvetica','B',13)
    pdf.set_fill_color(232,234,246)
    pdf.cell(190,8,'  Patient Information',fill=True,ln=True)
    pdf.set_font('Helvetica','',11); pdf.ln(2)
    pdf.cell(95,7,f'  Name   : {name}',ln=False)
    pdf.cell(95,7,f'Age    : {age} years',ln=True)
    pdf.cell(95,7,f'  Gender : {gender}',ln=False)
    pdf.cell(95,7,f'Date   : {datetime.date.today().strftime("%d-%m-%Y")}',ln=True)
    pdf.ln(4)

    # Stress Result
    sc_rgb = {0:(46,125,50),1:(249,168,37),2:(239,108,0),3:(198,40,40),4:(183,28,28)}
    r,g,b = sc_rgb[stress_level]
    pdf.set_font('Helvetica','B',13)
    pdf.set_fill_color(232,234,246)
    pdf.cell(190,8,'  Stress Diagnosis',fill=True,ln=True)
    pdf.ln(2)
    pdf.set_fill_color(r,g,b)
    pdf.set_text_color(255,255,255)
    pdf.set_font('Helvetica','B',14)
    info = STRESS_INFO[stress_level]
    pdf.cell(190,12,f'  Stress Level {stress_level} - {info["label"]}',fill=True,ln=True)
    pdf.set_text_color(0,0,0); pdf.ln(2)
    pdf.set_font('Helvetica','',10)
    pdf.multi_cell(190,6,f'  {clean_text(info["description"])}')
    pdf.ln(2)
    
    # Probability table
    pdf.set_font('Helvetica','B',11)
    pdf.cell(190,7,'  Model Confidence (Probability per Stress Level):',ln=True)
    pdf.set_font('Helvetica','',10)
    level_names = ['Low','Low-Medium','Medium','Medium-High','High']
    for i,(lname,prob) in enumerate(zip(level_names,probas)):
        marker = '<< Predicted' if i==stress_level else ''
        bar = '#' * int(prob*30) + '.' * (30-int(prob*30))
        pdf.cell(190,5.5,f'  Level {i} ({lname}): {bar}  {prob*100:.1f}% {marker}',ln=True)
    pdf.ln(3)

    # Health Score
    sl = 'Excellent' if health_score>=80 else 'Good' if health_score>=60 else 'Average' if health_score>=40 else 'Poor'
    pdf.set_font('Helvetica','B',12)
    pdf.cell(190,8,f'  Sleep Health Score: {health_score}/100  ({sl})',ln=True)
    pdf.ln(4)

    # Measurements table
    pdf.set_font('Helvetica','B',13)
    pdf.set_fill_color(232,234,246)
    pdf.cell(190,8,'  Measurements Table',fill=True,ln=True)
    pdf.ln(2)
    pdf.set_font('Helvetica','B',9)
    pdf.set_fill_color(200,210,240)
    for label,width in [('Measurement',60),('Your Value',35),('Normal Range',45),('Status',25),('Interpretation',25)]:
        pdf.cell(width,7,label,fill=True,border=1)
    pdf.ln()
    normal_map = [
        ('snoring_rate',    'Snoring Rate (%)',     '< 40%',       '< 40 = healthy airways'),
        ('respiration_rate','Respiration Rate',     '12-18 b/m',   'Calm, regular breathing'),
        ('body_temperature','Body Temp ( degreesF)',        '96.8-99 degreesF',   'Normal sleep temp'),
        ('limb_movement',  'Limb Movement',          '< 8',         'Low = deep sleep'),
        ('blood_oxygen',   'Blood Oxygen (%)',       '>= 95%',       'Vital for brain health'),
        ('rem_sleep',      'REM Sleep (%)',          '20-25%',      'Memory & mood recovery'),
        ('sleep_hours',    'Sleep Hours',            '7-9 hrs',     'WHO recommendation'),
        ('heart_rate',     'Heart Rate (BPM)',       '40-60 BPM',   'Low = healthy heart'),
    ]
    issue_keys = [i[0] for i in issues]
    pdf.set_font('Helvetica','',9)
    for key, label, norm, interp in normal_map:
        status = 'Abnormal' if key in issue_keys else 'Normal'
        fill   = (255,235,238) if key in issue_keys else (232,245,233)
        pdf.set_fill_color(*fill)
        pdf.cell(60,6,f'  {label}',     fill=True,border=1)
        pdf.cell(35,6,f'  {vals[key]:.1f}', fill=True,border=1)
        pdf.cell(45,6,f'  {norm}',      fill=True,border=1)
        pdf.cell(25,6,f'  {status}',    fill=True,border=1)
        pdf.cell(25,6,f'  {interp[:18]}',fill=True,border=1)
        pdf.ln()
    pdf.ln(4)

    # Abnormal
    if issues:
        pdf.set_font('Helvetica','B',13)
        pdf.set_fill_color(255,235,238)
        pdf.cell(190,8,'  Abnormal Readings & Why They Matter',fill=True,ln=True)
        pdf.set_font('Helvetica','',10); pdf.ln(1)
        for feat,direction,reason in issues:
            pdf.cell(190,6,f'  * {clean_text(feat)} is {clean_text(direction)} - {clean_text(reason)}',ln=True)
        pdf.ln(3)

    # Recommendations
    pdf.set_font('Helvetica','B',13)
    pdf.set_fill_color(227,242,253)
    pdf.cell(190,8,'  Doctor Recommendations',fill=True,ln=True)
    pdf.ln(2)
    for specialist,advice in recs:
        pdf.set_font('Helvetica','B',10)
        pdf.cell(190,6,f'  {clean_text(specialist)}',ln=True)
        pdf.set_font('Helvetica','',10)
        pdf.multi_cell(190,5.5,f'    {clean_text(advice)}')
        pdf.ln(1)

    # Tips
    pdf.ln(2)
    pdf.set_font('Helvetica','B',13)
    pdf.set_fill_color(232,245,233)
    pdf.cell(190,8,'  Lifestyle Tips to Reduce Stress',fill=True,ln=True)
    pdf.set_font('Helvetica','',10); pdf.ln(1)
    tips = [
        "Maintain a consistent sleep schedule (same bed/wake time every day).",
        "Avoid caffeine and heavy meals within 3 hours of bedtime.",
        "Keep the bedroom cool (65-68 degreesF), dark, and quiet.",
        "Practice 5-minute deep breathing before sleep to lower heart rate.",
        "Limit screen time 1 hour before sleep to improve REM quality.",
        "Exercise regularly but not within 2 hours of bedtime.",
    ]
    for tip in tips:
        pdf.cell(190,5.5,f'  * {tip}',ln=True)

    # Footer
    pdf.set_y(-18)
    pdf.set_fill_color(26,35,126)
    pdf.rect(0,pdf.get_y(),210,18,'F')
    pdf.set_text_color(255,255,255)
    pdf.set_font('Helvetica','I',9)
    pdf.cell(210,6,'This report is AI-generated for educational purposes only. Always consult a qualified medical professional.',align='C',ln=True)
    pdf.cell(210,6,'Student: Marium Ahmad | Final Year Project | SaYoPillow IEEE 2020 Dataset',align='C',ln=True)
    return bytes(pdf.output())

# ===============================================================================
# STREAMLIT UI
# ===============================================================================
st.set_page_config(page_title="Sleep Stress Analyzer", page_icon="", layout="wide")

st.markdown("""
<style>
.title-box{background:linear-gradient(135deg,#1a237e,#283593,#3949AB);
    padding:28px 20px;border-radius:16px;text-align:center;margin-bottom:24px;color:white;
    box-shadow:0 4px 15px rgba(26,35,126,0.4);}
.metric-card{background:#f8f9ff;border:1px solid #e0e4ff;border-radius:12px;
    padding:14px;text-align:center;margin:4px 0;}
.section-title{background:#e8eaf6;padding:10px 15px;border-radius:8px;
    font-weight:bold;margin:12px 0 8px 0;font-size:15px;border-left:4px solid #3949AB;}
.rec-box{background:#e3f2fd;padding:12px 15px;border-radius:8px;margin:6px 0;
    border-left:4px solid #1565c0;}
.issue-box{background:#fff3e0;padding:10px 15px;border-radius:8px;margin:5px 0;
    border-left:4px solid #e65100;}
.tip-box{background:#f1f8e9;padding:10px 15px;border-radius:8px;margin:5px 0;
    border-left:4px solid #558b2f;}
.insight-box{background:#fce4ec;padding:12px 15px;border-radius:10px;margin:8px 0;
    border-left:5px solid #c62828;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
  <h1 style="margin:0;font-size:2rem;"> Human Stress Detection System</h1>
  <p style="font-size:15px;margin:6px 0 2px;">Advanced Sleep Physiological Signal Analysis</p>
  <p style="font-size:12px;opacity:0.85;margin:0;">
    Final Year Project &nbsp;|&nbsp; Marium Ahmad &nbsp;|&nbsp; IEEE SaYoPillow Dataset (2020)
    &nbsp;|&nbsp; Voting Classifier (LR + RF + SVM)
  </p>
</div>
""", unsafe_allow_html=True)

# -- Sidebar info --------------------------------------------------------------
with st.sidebar:
    st.markdown("###  Quick Guide")
    st.markdown("""
**This app analyses 8 physiological signals collected during sleep to predict your stress level (0-4).**

**Input Signals:**
-  Snoring Rate
-  Respiration Rate  
-  Body Temperature
-  Limb Movement
-  Blood Oxygen
-  REM Sleep %
-  Sleep Hours
-  Heart Rate

**Stress Scale:**
- 0 = Low (Calm)
- 1 = Low-Medium
- 2 = Medium
- 3 = Medium-High
- 4 = High (Severe)
""")
    st.markdown("---")
    st.markdown("**Dataset:** 630 records, balanced across 5 stress levels")
    st.markdown("**Algorithm:** Voting Classifier ensemble")
    st.markdown("**Source:** IEEE Transactions on Consumer Electronics, 2020")

# -- Patient Info --------------------------------------------------------------
st.markdown('<div class="section-title"> Patient Information</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: patient_name   = st.text_input("Full Name", "Enter your name")
with c2: patient_age    = st.number_input("Age", 10, 90, 25)
with c3: patient_gender = st.selectbox("Gender", ["Female","Male","Other"])

# -- Input Sliders -------------------------------------------------------------
st.markdown('<div class="section-title"> Enter Your Sleep Measurements</div>', unsafe_allow_html=True)
st.caption(" Tip: The sliders use the actual data ranges from the IEEE dataset. Try different combinations to see how stress level changes.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("** Sleep & Breathing**")
    snoring_rate     = st.slider("Snoring Rate (%)\n  Normal: < 40%",    45.0, 100.0, 72.0, 0.5,
                                  help="Percentage of time snoring during sleep. High = airway problems")
    respiration_rate = st.slider("Respiration Rate (breaths/min)\n  Normal: 12-18", 
                                  16.0, 30.0, 22.0, 0.1,
                                  help="Breathing rate during sleep. Normal is 12-18")
    body_temperature = st.slider("Body Temperature ( degreesF)\n  Normal: 96.8-99", 
                                  85.0, 99.0, 92.0, 0.1,
                                  help="Core body temperature during sleep")
    rem_sleep        = st.slider("REM Sleep (%)\n  Normal: 20-25%",
                                  60.0, 105.0, 88.0, 0.5,
                                  help="Percentage of sleep spent in REM phase. Higher in dataset = more disturbed sleep cycles")

with col2:
    st.markdown("** Physiological Signals**")
    sleep_hours   = st.slider("Sleep Hours\n  Normal: 7-9 hrs",
                               0.0, 9.0, 3.5, 0.5,
                               help="Total sleep duration per night")
    limb_movement = st.slider("Limb Movement Rate\n  Normal: < 8",
                               4.0, 19.0, 12.0, 0.1,
                               help="How much your limbs move during sleep. High = restless sleep")
    blood_oxygen  = st.slider("Blood Oxygen (%)\n  Normal: >= 95%",
                               82.0, 97.0, 91.0, 0.1,
                               help="Blood oxygen saturation (SpO2). Below 95% is concerning")
    heart_rate    = st.slider("Heart Rate (BPM)\n  Normal: 40-60",
                               50.0, 85.0, 64.0, 0.5,
                               help="Resting heart rate during sleep. Lower = healthier")

vals = {
    'snoring_rate':snoring_rate,'respiration_rate':respiration_rate,
    'body_temperature':body_temperature,'limb_movement':limb_movement,
    'blood_oxygen':blood_oxygen,'rem_sleep':rem_sleep,
    'sleep_hours':sleep_hours,'heart_rate':heart_rate
}

st.markdown("")
_, mid, _ = st.columns([1,2,1])
with mid:
    analyse_btn = st.button(" Analyse My Stress Level", use_container_width=True, type="primary")

st.markdown("---")

# ==============================================================================
# RESULTS
# ==============================================================================
if analyse_btn:
    with st.spinner("Running analysis..."):
        stress_level = predict(vals)
        probas       = get_stress_proba(vals)
        health_score = calculate_sleep_health_score(vals)
        issues       = get_abnormal_readings(vals)
        recs         = get_doctor_recommendations(stress_level, issues)
        info         = STRESS_INFO[stress_level]

    # -- Result Banner ---------------------------------------------------------
    st.subheader(" Analysis Results")
    st.markdown(f"""
    <div style="background:{info['color']};border-left:8px solid {info['border']};
        color:{info['text']};padding:20px;border-radius:14px;font-size:1.3rem;
        font-weight:bold;margin-bottom:10px;">
        {info['emoji']} &nbsp; Predicted Stress Level: <b>{stress_level} - {info['label']}</b>
        <br><span style="font-size:0.85rem;font-weight:normal;">{info['description']}</span>
    </div>""", unsafe_allow_html=True)

    # Stress scale badges
    cols = st.columns(5)
    for i, (bc, lbl, clr) in enumerate(zip(cols,
            [' Low',' Low-Med',' Medium',' Med-High',' High'],
            ['#4CAF50','#CDDC39','#FF9800','#F44336','#B71C1C'])):
        border = "3px solid black" if i==stress_level else "1px solid #ccc"
        fw     = "bold" if i==stress_level else "normal"
        size   = "13px" if i==stress_level else "11px"
        bc.markdown(f"""<div style="background:{clr};color:white;text-align:center;
            padding:10px 2px;border-radius:8px;border:{border};
            font-weight:{fw};font-size:{size};">{lbl}</div>""", unsafe_allow_html=True)

    st.markdown("")

    # -- Summary Metrics -------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        sc_color = '#2e7d32' if health_score>=80 else '#e65100' if health_score>=50 else '#b71c1c'
        sc_label = 'Excellent ' if health_score>=80 else 'Good ' if health_score>=60 else 'Average (!)' if health_score>=40 else 'Poor '
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:12px;color:gray;">Sleep Health Score</div>
            <div style="font-size:36px;font-weight:bold;color:{sc_color};">{health_score}</div>
            <div style="font-size:11px;color:{sc_color};">{sc_label} / 100</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        conf = f"{max(probas)*100:.1f}%"
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:12px;color:gray;">Model Confidence</div>
            <div style="font-size:36px;font-weight:bold;color:#1565C0;">{conf}</div>
            <div style="font-size:11px;color:#1565C0;">Certainty of prediction</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        normal_count = 8 - len(issues)
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:12px;color:gray;">Normal Readings</div>
            <div style="font-size:36px;font-weight:bold;color:#2e7d32;">{normal_count}</div>
            <div style="font-size:11px;color:#2e7d32;">out of 8 signals</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:12px;color:gray;">Abnormal Signals</div>
            <div style="font-size:36px;font-weight:bold;color:#c62828;">{len(issues)}</div>
            <div style="font-size:11px;color:#c62828;">need attention</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # -- Charts Section --------------------------------------------------------
    st.markdown('<div class="section-title"> Detailed Visual Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([" Probability", " Gauge Charts", " Trend Analysis", " Population Comparison"])

    with tab1:
        c_left, c_right = st.columns([3,2])
        with c_left:
            fig_prob = plot_stress_probability(probas)
            st.pyplot(fig_prob, use_container_width=True)
            plt.close()
        with c_right:
            st.markdown("**What this chart shows:**")
            st.markdown("The model calculates a probability for each stress level. The highest bar is the predicted level.")
            for i, (lname, prob) in enumerate(zip(['Low','Low-Med','Medium','Med-High','High'], probas)):
                marker = " <- **Predicted**" if i == stress_level else ""
                st.markdown(f"Level {i} ({lname}): **{prob*100:.1f}%**{marker}")
        
        fi_fig = plot_feature_importance()
        if fi_fig:
            st.markdown("---")
            st.markdown("**Which signals matter most for stress prediction?**")
            st.pyplot(fi_fig, use_container_width=True)
            plt.close()

    with tab2:
        st.markdown("**Each bar shows your value vs the normal (green) and risk (red) zones:**")
        fig_gauge = plot_feature_gauge(vals, issues)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close()
        
        # Table summary
        st.markdown("** Readings Summary Table:**")
        table_data = []
        for key, (title, unit, vmin, vmax, norm_lo, norm_hi) in {
            'snoring_rate':     ('Snoring Rate',    '%',    45, 100, 0,    40),
            'respiration_rate': ('Respiration Rate','b/m',  16,  30, 12,   18),
            'body_temperature': ('Body Temp',       ' degreesF',   85,  99, 96.8, 99),
            'limb_movement':    ('Limb Movement',   '',      4,  19,  0,    8),
            'blood_oxygen':     ('Blood Oxygen',    '%',    82,  97, 95,  100),
            'rem_sleep':        ('REM Sleep',       '%',    60, 105, 20,   25),
            'sleep_hours':      ('Sleep Hours',     'hrs',   0,   9,  7,    9),
            'heart_rate':       ('Heart Rate',      'BPM',  50,  85, 40,   60),
        }.items():
            in_normal = norm_lo <= vals[key] <= norm_hi
            table_data.append({
                'Signal': f'{title} ({unit})',
                'Your Value': f'{vals[key]:.1f}',
                'Normal Range': f'{norm_lo}-{norm_hi}',
                'Status': ' Normal' if in_normal else '(!) Abnormal'
            })
        st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

    with tab3:
        st.markdown("**These charts simulate: what would happen to your stress if one factor changes?**")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_sleep = plot_stress_trend(vals)
            st.pyplot(fig_sleep, use_container_width=True)
            plt.close()
            st.caption(" The dashed line is your current sleep. Move it right to see stress drop.")
        with col_t2:
            fig_oxy = plot_oxygen_trend(vals)
            st.pyplot(fig_oxy, use_container_width=True)
            plt.close()
            st.caption(" Higher blood oxygen -> lower predicted stress. Red line = your current SpO2.")
        
        # Key insight
        ideal_vals = vals.copy()
        ideal_vals['sleep_hours'] = 8.0
        ideal_vals['blood_oxygen'] = 96.0
        ideal_stress = predict(ideal_vals)
        if ideal_stress < stress_level:
            improvement = stress_level - ideal_stress
            st.markdown(f"""
            <div class="insight-box">
             <b>Key Insight:</b> If you increased sleep to 8 hours and improved blood oxygen to 96%,
            your predicted stress could drop from <b>Level {stress_level} ({info['label']})</b> 
            to <b>Level {ideal_stress} ({STRESS_INFO[ideal_stress]['label']})</b> 
            - a <b>{improvement} level improvement!</b>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown(f"**How do your values compare to others with the same stress level ({info['label']})?**")
        fig_pop = plot_population_comparison(vals, stress_level)
        st.pyplot(fig_pop, use_container_width=True)
        plt.close()
        st.caption("The histogram shows the distribution of people with your stress level. The dashed line is you.")
        
        same_group = df_data[df_data['stress_level'] == stress_level]
        st.markdown("**Your position in the dataset:**")
        pc1, pc2, pc3, pc4 = st.columns(4)
        for col_ui, feat, label in zip([pc1,pc2,pc3,pc4],
                ['snoring_rate','sleep_hours','blood_oxygen','heart_rate'],
                ['Snoring Rate','Sleep Hours','Blood O2','Heart Rate']):
            percentile = (same_group[feat] < vals[feat]).mean() * 100
            col_ui.metric(label, f'{vals[feat]:.1f}', f'{percentile:.0f}th percentile')

    st.markdown("---")

    # -- Abnormal Readings ------------------------------------------------------
    st.markdown('<div class="section-title">(!) Abnormal Readings & Clinical Meaning</div>', unsafe_allow_html=True)
    if issues:
        for feat, direction, reason in issues:
            icon = "[HIGH]" if direction == "HIGH" else "[LOW]"
            st.markdown(f'<div class="issue-box">{icon} <b>{feat}</b> is <b>{direction}</b> - {reason}</div>',
                        unsafe_allow_html=True)
    else:
        st.success(" All 8 readings are within normal range! Your body shows healthy sleep patterns.")

    # -- Doctor Recommendations ------------------------------------------------
    st.markdown('<div class="section-title"> Personalized Medical Recommendations</div>', unsafe_allow_html=True)
    for specialist, advice in recs:
        st.markdown(f'<div class="rec-box"><b>{specialist}</b><br><span style="font-size:13px;">{advice}</span></div>',
                    unsafe_allow_html=True)

    # -- Lifestyle Tips --------------------------------------------------------
    st.markdown('<div class="section-title"> Evidence-Based Lifestyle Tips to Reduce Stress</div>', unsafe_allow_html=True)
    tips = [
        (" Consistent Schedule",   "Sleep and wake at the same time daily - even weekends. This regulates your circadian rhythm."),
        (" No Screens Before Bed",  "Avoid phone/laptop 1 hour before sleep. Blue light suppresses melatonin and reduces REM sleep."),
        (" Exercise Regularly",     "30 minutes of moderate exercise reduces cortisol. Avoid intense workouts 2 hours before bed."),
        (" Limit Caffeine",         "No caffeine after 2 PM. Caffeine has a 5-6 hour half-life and can disrupt deep sleep stages."),
        (" Deep Breathing",         "5 minutes of slow belly breathing before sleep lowers heart rate and activates the parasympathetic system."),
        (" Cool Room",              "Keep bedroom at 65-68 degreesF (18-20 degreesC). Core body temperature must drop 1-2 degreesF to initiate sleep."),
    ]
    tc1, tc2 = st.columns(2)
    for i, (title, tip) in enumerate(tips):
        col = tc1 if i % 2 == 0 else tc2
        col.markdown(f'<div class="tip-box"><b>{title}</b><br><span style="font-size:12px;">{tip}</span></div>',
                     unsafe_allow_html=True)

    # -- PDF Download ----------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-title"> Download Full Report (PDF)</div>', unsafe_allow_html=True)
    pdf_bytes = generate_pdf(patient_name, patient_age, patient_gender,
                              vals, stress_level, health_score, issues, recs, probas)
    _, dl_col, _ = st.columns([1,2,1])
    with dl_col:
        st.download_button(
            label=" Download Complete PDF Report",
            data=pdf_bytes,
            file_name=f"StressReport_{patient_name.replace(' ','_')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    st.info(" PDF includes: patient info, stress diagnosis, confidence probabilities, full measurements table, abnormal readings, doctor recommendations & lifestyle tips.")

# -- Footer About --------------------------------------------------------------
st.markdown("---")
with st.expander("i About This Project"):
    st.markdown("""
    ##  Human Stress Detection Based on Sleeping Habits
    
    **Dataset:** SaYoPillow - IEEE Transactions on Consumer Electronics (2020)  
    Collected using Smart Yoga Pillow sensors monitoring 8 physiological signals during sleep.
    
    **Why Sleep Data?**  
    Stress activates the sympathetic nervous system, causing measurable changes in:
    snoring, respiration rate, blood oxygen, heart rate, limb movement, and sleep architecture.
    
    **Model Pipeline:**
    - Data Cleaning -> EDA -> Feature Engineering (6 new features) -> Standardization -> Voting Classifier
    - Voting Classifier = Logistic Regression + Random Forest + SVM (soft voting)
    - Hyperparameter tuning via GridSearchCV
    
    **Advanced Features in This App:**
    - 5-class stress prediction with probability distribution
    - Feature gauge charts (normal vs risk zone)
    - Trend simulation (what-if analysis)
    - Population comparison charts
    - Feature importance visualization
    - Sleep health score (0-100)
    - Medical recommendations + lifestyle tips
    - Downloadable PDF report with full analysis
    
    **Student:** Marium Ahmad | **Project:** Final Year - Trend & Predictive Analysis
    """)
